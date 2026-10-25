"""Main chat engine (Pro) with an awaited Flash retrieval sub-agent.

Adapted from ``work/generated-memory-demo/src/lettaproto/chat.py``
(sha256 aa69dfea9e3fb5d22006b38107e1a75b19a0347c0f2811ced2c26a05b3388607).

Differences:
* Context is the FULL published summary + this session's recent turns (built in
  ``context.py``), not ``idea.md``/``stable.md``.
* ``submit_memory_update`` is REMOVED, so the main chat can never write the old
  8788 worker or any memory directly. Memory changes only through the serial
  maintenance queue after a turn completes.
* On a successful answer the assistant row and its maintenance-queue row are
  persisted to this trial's ``chat_log`` / ``maint_queue`` in ONE transaction (so
  a crash cannot leave a saved answer without its queue row); the queue then runs
  the maintenance CLI asynchronously. The DSML leak guard and the single
  tools-disabled recovery turn are preserved verbatim in spirit.

Routing is explicit: every main-chat call passes ``model=config.main_chat_model``
(Pro). The delegated Memory Agent calls keep the default (``config.chat_model``,
Flash), so switching the main chat never upgrades retrieval.
"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

from . import agents
from .events import current, emit, observe
from .provider_stream import DraftGuard, StreamProtocolError
from .config import Config
from .context import build_context
from .deepseek import DeepSeekClient
from .retrieval import now_shanghai
from .summary_store import Published
from .util import looks_like_tool_serialization


class EmptyModelResponse(RuntimeError):
    """No complete, user-facing answer after the single bounded recovery turn."""


# Appended only to the tools-disabled recovery turn: must never re-emit internal markup.
RECOVERY_NUDGE = (
    "请直接用简体中文给出最终回答，基于上面的记忆与已有工具结果；"
    "不要再输出任何工具调用、内部标记或代码块。"
)


TOOL_PROTOCOL_NUDGE = (
    "上一轮正文不是有效回答，而是工具序列化文本，工具尚未因此执行。"
    "如果需要检索，请使用本请求 tools 中注册的原生 function/tool_calls 协议；"
    "不要用 XML、DSML 或普通正文模拟工具调用，不得假设已有工具结果。"
    "请复用已有上下文和真实工具结果；如果无需工具，请直接给出正常最终正文。"
)


def _valid_final_answer(text: str | None) -> bool:
    if not text or not text.strip():
        return False
    return not looks_like_tool_serialization(text)


def assistant_wire(content: str, tool_calls: list[dict[str, Any]] | None = None,
                   reasoning_content: Any = None,
                   response_items: Any = None) -> dict[str, Any]:
    """One assistant entry for the API's ``messages`` history.

    Thinking mode is on by default (effort ``high``), and DeepSeek requires the
    ``reasoning_content`` of every previous turn to be passed back VERBATIM on
    requests that carry ``tools`` (otherwise HTTP 400). It is copied through
    exactly as received, and only when the field is actually present, so payloads
    without it stay compatible. It is never merged into ``content``, logged, or
    exposed to the user.
    """
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    if isinstance(reasoning_content, str) and reasoning_content:
        message["reasoning_content"] = reasoning_content
    if isinstance(response_items, list):
        message["_response_items"] = response_items
    return message


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "delegate_memory",
            "description": (
                "Ask the independent Memory Agent to retrieve and verify specific facts from THIS "
                "trial's own past chats (strictly before now). Say what you already know and the "
                "specific gaps to fill; the agent chooses its own keywords/aliases/time range and "
                "returns status/findings with exact sources and message times."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "gaps": {"type": "array", "items": {"type": "string"},
                             "description": "Specific unknowns to resolve (required)."},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "entities": {"type": "array", "items": {"type": "string"}},
                    "time_hint": {"type": "string", "description": "Time range or relative hint."},
                    "known": {"type": "string", "description": "What you already know."},
                },
                "required": ["gaps"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Legacy alias of delegate_memory: delegate a lookup with a single query string.",
            "parameters": {"type": "object",
                           "properties": {"query": {"type": "string", "description": "What to look up"}},
                           "required": ["query"]},
        },
    },
]

AGENT_SYSTEM_ADDENDUM = (
    "\n\nYou have tools. `delegate_memory` asks the independent Memory Agent for PAST evidence from "
    "this user's own earlier chats (preferred): give the specific gaps plus any keywords/entities/time "
    "hint you suspect; WAIT for its result and use it. `search_memory` is a legacy alias taking a "
    "single query string. You cannot write memory yourself; a background process maintains the "
    "long-term summary. The current moment is the injected SIMULATED NOW and only history before it "
    "exists. Do not claim you saved or background-processed anything beyond what a tool result reports."
)


def _log(conn: sqlite3.Connection, instance: str, conv_id: str, role: str,
         content: str, sources: list[Any] | None = None, *, commit: bool = True,
         created_at: float | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO chat_log(created_at, instance, conversation_id, role, content, sources_json)"
        " VALUES(?,?,?,?,?,?)",
        (time.time() if created_at is None else created_at, instance, conv_id, role, content,
         json.dumps(sources or [], ensure_ascii=False)),
    )
    if commit:
        conn.commit()
    return cur.lastrowid


def _retrieval_payload(result: dict[str, Any]) -> dict[str, Any]:
    """Bounded, valid-JSON payload handed to the main model (no string slicing)."""
    ev = []
    for e in (result.get("evidence") or [])[:8]:
        text = e.get("text") or ""
        ev.append({"source_id": e.get("source_id"), "source": e.get("source"), "role": e.get("role"),
                   "message_at": e.get("message_at"), "text": text[:3000],
                   "truncated": len(text) > 3000 or bool(e.get("truncated"))})
    return {
        "status": result.get("status"),
        "findings": result.get("findings") or [],
        "evidence": ev,
        "gaps": result.get("gaps") or [],
        "expandable_source_ids": result.get("expandable") or [],
        "engine": result.get("engine"),
        "retrieval_rounds": result.get("rounds"),
        "note": result.get("note", ""),
        "time_note": "message_at = when the message was sent (Asia/Shanghai +08:00).",
    }


def _run_memory_tool(name, args, config, conn, published, instance, client,
                     now_epoch, user_row_id, conversation_id):
    if name not in ('search_memory', 'delegate_memory'):
        return None
    return agents.memory_agent_search(
        config, conn, published, str(args.get('query', '')) if name == 'search_memory' else '',
        instance=instance, client=client, now_epoch=now_epoch,
        delegation=args if name == 'delegate_memory' else None,
        exclude_chat_ids=(user_row_id,), conversation_id=conversation_id)


def chat_agentic(
    config: Config,
    conn: sqlite3.Connection,
    message: str,
    *,
    published: Published,
    instance: str = "trial",
    recent_limit: int | None = None,
    recent_token_budget: int | None = None,
    max_tokens: int | None = None,
    conversation_id: str | None = None,
    maintenance: Any | None = None,
    stream: bool = False,
    client: Any | None = None,
    retrieval_client: Any | None = None,
    now_epoch: float | None = None,
    allow_memory_tools: bool = True,
) -> dict[str, Any]:
    """Pro main chat: full published summary + this session's recent context, tool loop."""
    if max_tokens is None:
        max_tokens = config.chat_max_output_tokens
    now_epoch = time.time() if now_epoch is None else now_epoch
    now = now_shanghai(now_epoch)
    emit('context_loading')
    bundle = build_context(
        published, conn, instance=instance,
        recent_limit=recent_limit if recent_limit is not None else config.default_recent_messages,
        recent_token_budget=(recent_token_budget if recent_token_budget is not None
                             else config.default_recent_token_budget),
        now=now, conversation_id=conversation_id,
    )
    client = client if client is not None else DeepSeekClient(config)
    retrieval_client = retrieval_client if retrieval_client is not None else client
    conv_id = conversation_id or f"trial-{int(now_epoch * 1000)}"
    # log the user turn first so the model can cite its real chat#N id as a source
    user_row_id = _log(conn, instance, conv_id, "user", message, created_at=now_epoch)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": bundle.system_prompt()
         + (AGENT_SYSTEM_ADDENDUM if allow_memory_tools else "")},
        {"role": "user", "content": f"(message id: chat#{user_row_id}; sent_at {now})\n{message}"},
    ]
    emit('context_loaded', stats=bundle.stats, memory_version=published.version)

    def complete(history, *, tools=None, recovery=False, tokens=max_tokens):
        emit('model_started', model=config.main_chat_model, recovery=recovery,
             tools_enabled=bool(tools))
        if not stream:
            return client.chat(history, max_tokens=tokens, temperature=0.6,
                               tools=tools, model=config.main_chat_model)
        guard = DraftGuard(emit)
        def on_event(kind, **data):
            if kind == 'content':
                guard.feed(data['text'])
            else:
                emit('model_thinking', **data)
        try:
            result = client.chat_stream(history, max_tokens=tokens, temperature=0.6,
                                        tools=tools, model=config.main_chat_model,
                                        on_event=on_event)
        except StreamProtocolError:
            emit('draft_reset', reason='provider_stream_error')
            raise
        choice = result['choices'][0]
        if choice['finish_reason'] == 'stop' and not choice['message'].get('tool_calls'):
            guard.feed('', final=True)
        else:
            emit('draft_reset', reason=choice['finish_reason'])
        emit('model_finished', finish_reason=choice['finish_reason'], usage=result.get('usage'),
             usage_missing=not bool(result.get('usage')))
        return result

    trace: list[dict[str, Any]] = []
    answer = ""
    rounds = 0
    limit = config.chat_tool_round_limit
    protocol_corrected = False
    while rounds < limit:
        rounds += 1
        try:
            payload = complete(messages, tools=TOOLS if allow_memory_tools else None)
        except StreamProtocolError:
            trace.append({"tool": "invalid_final", "reason": "provider_stream_error", "round": rounds})
            break
        msg = client.message(payload)
        calls = msg.get("tool_calls") or []
        finish = (payload.get("choices") or [{}])[0].get("finish_reason")
        if finish not in (None, "stop", "tool_calls"):
            trace.append({"tool": "invalid_final", "reason": finish, "round": rounds})
            break
        if calls and not allow_memory_tools:
            trace.append({"tool": "invalid_final", "reason": "tools_disabled", "round": rounds})
            break
        if not calls:
            candidate = msg.get("content") or ""
            if (allow_memory_tools and looks_like_tool_serialization(candidate) and not protocol_corrected
                    and rounds < limit):
                # Consume the NEXT existing tool round, never interpret the text
                # as an action or insert it as an assistant/tool history entry.
                protocol_corrected = True
                trace.append({'tool': 'protocol_correction', 'round': rounds})
                emit('draft_reset', reason='tool_protocol_correction')
                messages.append({'role': 'user', 'content': TOOL_PROTOCOL_NUDGE})
                continue
            if finish == "tool_calls":
                trace.append({"tool": "invalid_final", "reason": "missing_tool_calls", "round": rounds})
                answer = ""
            elif not _valid_final_answer(candidate):
                trace.append({"tool": "invalid_final", "reason": "tool_serialization", "round": rounds})
                answer = ""
            else:
                answer = candidate
            break
        assistant_content = msg.get("content") or ""
        if looks_like_tool_serialization(assistant_content):
            assistant_content = ""
        messages.append(assistant_wire(assistant_content, calls, msg.get("reasoning_content"),
                                       msg.get("_response_items")))
        for call in calls:
            name = (call.get("function") or {}).get("name")
            try:
                args = json.loads((call.get("function") or {}).get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            args = args if isinstance(args, dict) else {}
            parent = current()
            tool_observer = parent.child(tool_call_id=call.get('id')) if parent else None
            if tool_observer:
                tool_observer.emit('tool_call', name=name, arguments=args,
                                   arguments_json=(call.get('function') or {}).get('arguments'))
            with observe(tool_observer):
                result = _run_memory_tool(
                    name, args, config, conn, published, instance, retrieval_client, now_epoch,
                    user_row_id, conversation_id)
            if result is not None:
                trace.append({
                    "tool": name, "query": result.get("query"),
                    "status": result.get("status"), "sufficient": result.get("sufficient"),
                    "sources": result.get("sources"), "rounds": result.get("rounds"),
                    "evidence": len(result.get("evidence") or []),
                    "engine": result.get("engine"), "findings": result.get("findings"),
                })
                content = json.dumps(_retrieval_payload(result), ensure_ascii=False)
            else:
                content = json.dumps({"error": f"unknown tool {name}"}, ensure_ascii=False)
            if tool_observer:
                # The exact string appended to messages, with the same existing evidence bounds.
                tool_observer.emit('retrieval_handoff', name=name, content=content,
                                   bounds={'evidence_items': 8, 'evidence_chars': 3000,
                                           'expand': 'read_message / expand; truncated flags retained'})
            messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": content})
            emit('pro_resuming', tool_call_id=call.get('id'))
    if not _valid_final_answer(answer):
        recovery = messages + [{"role": "user", "content": RECOVERY_NUDGE}]
        emit('draft_reset', reason='recovery')
        payload = complete(recovery, recovery=True,
                           tokens=max(max_tokens, config.chat_final_max_output_tokens))
        answer = client.content(payload)
        finish = (payload.get("choices") or [{}])[0].get("finish_reason")
        trace.append({"tool": "final_answer", "rounds": rounds, "limit": limit, "finish_reason": finish})
        if (not _valid_final_answer(answer) or finish not in (None, "stop")
                or client.message(payload).get('tool_calls')):
            raise EmptyModelResponse(f"incomplete final answer (finish={finish})")

    # Atomic: the assistant message and its maintenance-queue row commit in ONE
    # transaction. A crash can therefore never leave a saved answer without its
    # queue row (and never enqueue a turn that was not saved).
    try:
        assistant_row_id = _log(conn, instance, conv_id, "assistant", answer, commit=False,
                                created_at=now_epoch)
        queued = None
        if maintenance is not None:
            queued = maintenance.enqueue_turn(user_row_id, assistant_row_id, conn=conn)
        if queued and hasattr(maintenance, 'commit_turn'):
            maintenance.commit_turn(conn, queued, current(), conv_id)
        else:
            conn.commit()
    except BaseException:
        conn.rollback()
        raise
    if queued:
        emit('maintenance_queued', **queued)
    emit('answer_committed', assistant_message_id=f'chat#{assistant_row_id}')
    return {
        "conversation_id": conv_id,
        "user_message_id": f"chat#{user_row_id}",
        "assistant_message_id": f"chat#{assistant_row_id}",
        "answer": answer,
        "tool_trace": trace,
        "tool_rounds": rounds,
        "now": now,
        "agent_version": agents.AGENT_VERSION,
        "context_stats": bundle.stats,
        "memory": {"present": published.present, "version": published.version},
        "maintenance": queued,
        "api_usage_total": client.usage.to_dict(),
    }
