"""Chat engines.

  direct : build context (full idea.md) and call DeepSeek; fully observable.
  letta  : sync idea.md into the Letta agent MemFS and chat via the Letta CLI.
"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

from . import db, ideastore
from . import agents
from .config import Config
from .context import build_context
from .deepseek import DeepSeekClient
from .letta import chat_cli, write_agent_memory
from .retrieval import now_shanghai
from .util import looks_like_tool_serialization


class EmptyModelResponse(RuntimeError):
    """No complete, user-facing answer after the single bounded recovery turn."""


# Appended only to the tools-disabled recovery turn: it must never re-emit internal markup.
RECOVERY_NUDGE = (
    "请直接用简体中文给出最终回答，基于上面的记忆与已有工具结果；"
    "不要再输出任何工具调用、内部标记或代码块。"
)


def _valid_final_answer(text: str | None) -> bool:
    """A user-facing answer must be non-empty and free of leaked tool serialization."""
    if not text or not text.strip():
        return False
    return not looks_like_tool_serialization(text)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "delegate_memory",
            "description": (
                "Ask the independent Memory Agent to retrieve and verify specific facts from the "
                "user's PAST history (strictly before now). Say what you already know and the specific "
                "gaps to fill; the agent chooses its own keywords/aliases/time range and returns "
                "status/findings with exact sources and message times."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "gaps": {"type": "array", "items": {"type": "string"},
                             "description": "Specific unknowns to resolve (required)."},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "entities": {"type": "array", "items": {"type": "string"}},
                    "time_hint": {"type": "string", "description": "Time range or relative hint."},
                    "known": {"type": "string", "description": "What you already know from memory/session."},
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
    {
        "type": "function",
        "function": {
            "name": "submit_memory_update",
            "description": (
                "Suggest a durable memory update for the background Memory Agent to verify and "
                "possibly save. You cannot write memory yourself. Provide the REAL user message "
                "ids (e.g. chat#12) that support it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string", "description": "The proposed memory, in the user's terms"},
                    "source_message_ids": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                },
                "required": ["statement", "source_message_ids"],
            },
        },
    },
]

AGENT_SYSTEM_ADDENDUM = (
    "\n\nYou have tools. `delegate_memory` asks the independent Memory Agent for PAST evidence "
    "(preferred): give the specific gaps, plus any keywords/entities/time hint you suspect; WAIT for "
    "its result and use it. `search_memory` is a legacy alias taking a single query string. "
    "`submit_memory_update` queues a durable-memory suggestion for the background Memory Agent, with "
    "REAL user message ids (e.g. chat#12); you never write memory directly. The current moment is the "
    "injected SIMULATED NOW and only history before it exists. Do not claim you saved or "
    "background-processed anything beyond what a tool result reports."
)


def _log(conn: sqlite3.Connection, instance: str, conv_id: str, role: str, content: str, sources: list[Any] | None = None) -> int:
    import json

    cur = conn.execute(
        "INSERT INTO chat_log(created_at, instance, conversation_id, role, content, sources_json) VALUES(?,?,?,?,?,?)",
        (time.time(), instance, conv_id, role, content, json.dumps(sources or [], ensure_ascii=False)),
    )
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
        "time_note": "message_at = when the message was sent (Asia/Shanghai +08:00). "
                     "event_at (if present) = when the described event happened; may be null.",
    }


def chat_direct(
    config: Config,
    conn: sqlite3.Connection,
    message: str,
    *,
    instance: str = "prod",
    recent_limit: int | None = None,
    recent_token_budget: int | None = None,
    from_chatlog: bool = True,
    log: bool = True,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    bundle = build_context(
        config, conn, instance=instance, recent_limit=recent_limit,
        recent_token_budget=recent_token_budget, from_chatlog=from_chatlog, now=now_shanghai(),
    )
    client = DeepSeekClient(config)
    messages = [{"role": "system", "content": bundle.system_prompt()}, {"role": "user", "content": message}]
    payload = client.chat(messages, max_tokens=max_tokens, temperature=0.6, model=config.main_chat_model)
    answer = DeepSeekClient.content(payload)
    usage = payload.get("usage") or {}
    conv_id = f"proto-{int(time.time())}"
    if log:
        _log(conn, instance, conv_id, "user", message)
        _log(conn, instance, conv_id, "assistant", answer)
    return {
        "conversation_id": conv_id,
        "answer": answer,
        "context_stats": bundle.stats,
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        },
        "api_usage_total": client.usage.to_dict(),
    }


def chat_letta(
    config: Config,
    conn: sqlite3.Connection,
    message: str,
    agent_id: str,
    *,
    instance: str = "prod",
    sync_memory: bool = True,
    new_conversation: bool = True,
) -> dict[str, Any]:
    if sync_memory:
        sync = write_agent_memory(
            config, agent_id, ideastore.load(config, instance), commit_message="sync: idea.md -> agent memory"
        )
        if not sync.get("ok"):
            # never answer from a stale/partial agent memory
            raise RuntimeError(f"memory sync to Letta agent failed; aborting chat: {sync}")
    data = chat_cli(config, agent_id, message, new_conversation=new_conversation)
    if data.get("result"):
        _log(conn, instance, data.get("conversation_id") or "letta", "user", message)
        _log(conn, instance, data.get("conversation_id") or "letta", "assistant", data["result"])
    return data


def chat_agentic(
    config: Config,
    conn: sqlite3.Connection,
    message: str,
    *,
    instance: str = "prod",
    recent_limit: int | None = None,
    recent_token_budget: int | None = None,
    from_chatlog: bool = True,
    worker: Any | None = None,
    max_tokens: int | None = None,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    """Main chat model: full idea.md + recent context, with a real tool-call loop.

    delegate_memory / search_memory -> delegated to the independent Memory Agent (awaited).
    submit_memory_update -> persisted to the background queue (asynchronous).
    """
    if max_tokens is None:
        max_tokens = config.chat_max_output_tokens
    now_epoch = time.time()
    now = now_shanghai(now_epoch)
    bundle = build_context(
        config, conn, instance=instance, recent_limit=recent_limit,
        recent_token_budget=recent_token_budget, from_chatlog=from_chatlog, now=now,
        conversation_id=conversation_id,
    )
    client = DeepSeekClient(config)
    conv_id = conversation_id or f"proto-{int(time.time() * 1000)}"
    # log the user turn first so the model can cite its real chat#N id as a source
    user_row_id = _log(conn, instance, conv_id, "user", message)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": bundle.system_prompt() + AGENT_SYSTEM_ADDENDUM},
        {"role": "user", "content": f"(message id: chat#{user_row_id}; sent_at {now})\n{message}"},
    ]
    trace: list[dict[str, Any]] = []
    answer = ""
    rounds = 0
    limit = config.chat_tool_round_limit
    while rounds < limit:
        rounds += 1
        payload = client.chat(messages, max_tokens=max_tokens, temperature=0.6, tools=TOOLS,
                              model=config.main_chat_model)
        msg = client.message(payload)
        calls = msg.get("tool_calls") or []
        finish = (payload.get("choices") or [{}])[0].get("finish_reason")
        if not calls:
            candidate = msg.get("content") or ""
            # A round with no structured tool call is only a final answer if it is a
            # complete, user-facing string. Truncated (finish=length) or leaked
            # tool-serialization content is rejected here and resolved by the single
            # tools-disabled recovery turn below.
            if finish == "length":
                trace.append({"tool": "invalid_final", "reason": "length", "round": rounds})
                answer = ""
            elif not _valid_final_answer(candidate):
                trace.append({"tool": "invalid_final", "reason": "tool_serialization", "round": rounds})
                answer = ""
            else:
                answer = candidate
            break
        # Real structured tool calls keep the loop going. Never replay leaked markup
        # as assistant context.
        assistant_content = msg.get("content") or ""
        if looks_like_tool_serialization(assistant_content):
            assistant_content = ""
        messages.append({"role": "assistant", "content": assistant_content, "tool_calls": calls})
        for call in calls:
            name = (call.get("function") or {}).get("name")
            try:
                args = json.loads((call.get("function") or {}).get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            if name == "search_memory":
                result = agents.memory_agent_search(
                    config, conn, str(args.get("query", "")), instance=instance, client=client,
                    now_epoch=now_epoch, exclude_chat_ids=(user_row_id,), conversation_id=conversation_id,
                )
            elif name == "delegate_memory":
                result = agents.memory_agent_search(
                    config, conn, "", instance=instance, client=client,
                    now_epoch=now_epoch, delegation=args, exclude_chat_ids=(user_row_id,), conversation_id=conversation_id,
                )
            else:
                result = None
            if result is not None:
                trace.append({
                    "tool": name, "query": result.get("query"),
                    "status": result.get("status"), "sufficient": result.get("sufficient"),
                    "sources": result.get("sources"), "rounds": result.get("rounds"),
                    "evidence": len(result.get("evidence") or []),
                    "engine": result.get("engine"), "findings": result.get("findings"),
                })
                content = json.dumps(_retrieval_payload(result), ensure_ascii=False)
            elif name == "submit_memory_update":
                spec = {"statement": args.get("statement", ""), "reason": args.get("reason", "")}
                enq = _enqueue(config, conn, worker, instance, "suggestion", spec, args.get("source_message_ids") or [])
                trace.append({"tool": "submit_memory_update", "task": enq, "statement": args.get("statement")})
                content = json.dumps({"queued": True, **enq}, ensure_ascii=False)
            else:
                content = json.dumps({"error": f"unknown tool {name}"}, ensure_ascii=False)
            messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": content})
    if not _valid_final_answer(answer):
        # Single bounded recovery turn: tools disabled so it cannot emit another tool
        # call; every tool result gathered above is preserved in `messages`.
        recovery = messages + [{"role": "user", "content": RECOVERY_NUDGE}]
        payload = client.chat(
            recovery, max_tokens=max(max_tokens, config.chat_final_max_output_tokens), temperature=0.6,
            model=config.main_chat_model,
        )
        answer = client.content(payload)
        finish = (payload.get("choices") or [{}])[0].get("finish_reason")
        trace.append({"tool": "final_answer", "rounds": rounds, "limit": limit, "finish_reason": finish})
        if not _valid_final_answer(answer) or finish == "length":
            # give up clearly: no bad answer is returned or written to history
            raise EmptyModelResponse(f"incomplete final answer (finish={finish})")

    _log(conn, instance, conv_id, "assistant", answer)
    return {
        "conversation_id": conv_id,
        "user_message_id": f"chat#{user_row_id}",
        "answer": answer,
        "tool_trace": trace,
        "tool_rounds": rounds,
        "now": now,
        "agent_version": agents.AGENT_VERSION,
        "context_stats": bundle.stats,
        "api_usage_total": client.usage.to_dict(),
    }


def _enqueue(config, conn, worker, instance, kind, spec, source_ids) -> dict[str, Any]:
    if worker is not None:
        return worker.enqueue(instance, kind, spec, source_ids)
    from .worker import MemoryWorker  # local import avoids a hard dependency for direct CLI use

    return MemoryWorker(config).enqueue(instance, kind, spec, source_ids)
