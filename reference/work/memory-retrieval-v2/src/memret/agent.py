"""The independent Memory Agent: a separate Flash role with its own tool loop.

It receives a STRUCTURED delegation (current question, bounded session context,
bounded memory excerpt, explicit gaps/keywords/entities/time hint, simulated now),
chooses its own searches, reads/expands, and ends with a reserved synthesis turn.
The harness then materialises the exact original evidence for the cited sources
(so no model has to re-type quotes and nothing is dropped by string slicing).
"""
from __future__ import annotations

import json

from . import lm
from . import prompts as P
from . import settings as S
from . import store
from . import tools as T
from .lm import extract_json

AGENT_SEARCH_TOOL = {"type": "function", "function": {
    "name": "search_history",
    "description": "Ranked search over the user's PAST messages (strictly before now).",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string", "description": "Natural query phrase."},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "entities": {"type": "array", "items": {"type": "string"}, "description": "People/aliases."},
        "time_start": {"type": "string", "description": "ISO lower bound (optional)."},
        "time_end": {"type": "string", "description": "ISO upper bound (optional, < now)."},
        "conversation_id": {"type": "string", "description": "Restrict to one conversation."},
        "limit": {"type": "integer"}}, "required": ["query"]}}}
AGENT_READ_TOOL = {"type": "function", "function": {
    "name": "read_message",
    "description": "Read the full original text of one past message by source_id.",
    "parameters": {"type": "object", "properties": {
        "source_id": {"type": "string"}, "conversation_id": {"type": "string"},
        "max_chars": {"type": "integer"}}, "required": ["source_id"]}}}
AGENT_EXPAND_TOOL = {"type": "function", "function": {
    "name": "expand",
    "description": "Expand a message into its same-conversation neighbours (target included).",
    "parameters": {"type": "object", "properties": {
        "source_id": {"type": "string"}, "conversation_id": {"type": "string"},
        "before": {"type": "integer"}, "after": {"type": "integer"},
        "offset": {"type": "integer"}}, "required": ["source_id"]}}}
AGENT_TOOLS = [AGENT_SEARCH_TOOL, AGENT_READ_TOOL, AGENT_EXPAND_TOOL]


def _delegation_text(delg: dict) -> str:
    ctx = delg.get("session_context") or []
    ctx_txt = "\n".join(f'[{m["role"]}] (message_at {m.get("message_at")}) {m["text"]}' for m in ctx) \
        or "(no earlier messages in this session)"
    mem = (delg.get("memory_excerpt") or "").strip() or "(no long-term memory excerpt supplied)"
    lines = [
        f'SIMULATED NOW: {delg["now_iso"]}  (timezone Asia/Shanghai, UTC+08:00). '
        f'History at or after this moment does not exist yet.',
        "",
        "=== CURRENT QUESTION (what the user just asked) ===",
        delg.get("question", ""),
        "",
        "=== CURRENT SESSION CONTEXT (bounded; each carries message_at) ===",
        ctx_txt,
        "",
        "=== KNOWN MEMORY EXCERPT (may be empty / unrelated; context only) ===",
        mem,
        "",
        "=== THE MAIN MODEL'S DELEGATION (its own request) ===",
        "gaps to resolve: " + json.dumps(delg.get("gaps") or [], ensure_ascii=False),
        "keywords: " + json.dumps(delg.get("keywords") or [], ensure_ascii=False),
        "entities: " + json.dumps(delg.get("entities") or [], ensure_ascii=False),
        "time hint: " + json.dumps(delg.get("time_hint") or "", ensure_ascii=False),
        "what the main model already knows: " + json.dumps(delg.get("known") or "", ensure_ascii=False),
    ]
    return "\n".join(lines)


def _dispatch(name: str, args: dict, bound: float, con) -> dict:
    if name == "search_history":
        return T.search_history(args, bound, con)
    if name == "read_message":
        return T.read_message(args, bound, con)
    if name == "expand":
        return T.expand(args, bound, con)
    return {"tool": name, "error": "unknown tool"}


def _collect_sources(result: dict, pool: dict) -> None:
    """Record every message the tools returned, keyed by source_id."""
    items = []
    if result.get("tool") == "search_history":
        items = result.get("results", [])
    elif result.get("tool") == "read_message":
        items = [result["item"]] if result.get("item") else []
    elif result.get("tool") == "expand":
        items = result.get("items", [])
    for it in items:
        sid = it.get("source_id")
        if not sid:
            continue
        slot = pool.setdefault(sid, {"source_id": sid, "message_id": it.get("message_id"),
                                     "conversation_id": it.get("conversation_id"),
                                     "node_id": it.get("node_id"), "role": it.get("role"),
                                     "message_at": it.get("message_at"), "time": it.get("time"),
                                     "text": it.get("text") or it.get("snippet") or "",
                                     "full": bool(it.get("text"))})
        if it.get("text") and (not slot.get("full")):
            slot["text"], slot["full"] = it["text"], True


def _materialise(ids: list[str], pool: dict, bound: float, con, limit: int = 14) -> list[dict]:
    """Exact original text for cited sources (re-read from DB, guarded)."""
    out, seen = [], set()
    for sid in ids or []:
        if not isinstance(sid, str) or sid in seen:
            continue
        seen.add(sid)
        rows = store.find_row(con, sid, bound)
        if not rows:
            continue
        r = rows[0]
        out.append({"source_id": store.sid(r), "message_id": r["id"], "conversation_id": r["conv"],
                    "role": r["role"], "message_at": S.now_iso(r["t"]), "time": S.utc_iso(r["t"]),
                    "text": r["text"]})
        if len(out) >= limit:
            break
    return out


def finalize(trace: dict, final: dict, pool: dict, bound: float, con) -> dict:
    """Apply the structured synthesis: drop unverified findings, resolve status,
    and materialise the exact original evidence for the cited sources."""
    trace["final_ok"] = True
    # accept a verbatim source_id, or be lenient about a cited message_id (exact/prefix)
    msg_map: dict[str, str] = {}
    for sid, v in pool.items():
        mid = v.get("message_id")
        if mid:
            msg_map.setdefault(mid, sid)

    def resolve(s: str) -> str | None:
        if s in pool:
            return s
        if s in msg_map:
            return msg_map[s]
        if len(s) >= 6:
            for mid, sid in msg_map.items():             # unique prefix
                if mid.startswith(s) or (s.startswith(mid) and len(mid) >= 6):
                    return sid
        return None

    findings = final.get("findings") or []
    clean = []
    for f in findings if isinstance(findings, list) else []:
        if not isinstance(f, dict):
            continue
        sids = []
        for s in (f.get("source_ids") or []):
            r = resolve(s) if isinstance(s, str) else None
            if r and r not in sids:
                sids.append(r)
        f = dict(f)
        f["source_ids"] = sids
        f["source_verified"] = bool(sids)
        clean.append(f)
    trace["findings"] = clean
    trace["gaps"] = final.get("gaps") or []
    trace["expandable"] = [s for s in (final.get("expandable_source_ids") or [])
                           if isinstance(s, str) and resolve(s)]
    trace["note"] = final.get("note", "")
    ev_ids = [resolve(s) for s in (final.get("evidence_source_ids") or []) if isinstance(s, str)]
    ev_ids = [s for s in ev_ids if s]
    trace["evidence"] = _materialise(ev_ids, pool, bound, con)
    verified = bool(trace["evidence"]) or any(f.get("source_verified") for f in clean)
    status = final.get("status")
    if status in ("found", "partial") and verified:
        trace["status"] = status
    elif status == "not_found":
        trace["status"] = "not_found"
    else:
        trace["status"] = "partial" if verified else "not_found"
    if trace["status"] == "not_found":
        trace["findings"] = []
    if trace["status"] in ("found", "partial") and not trace["evidence"]:
        fb = [s for f in trace["findings"] for s in (f.get("source_ids") or [])]
        trace["evidence"] = _materialise(fb, pool, bound, con)
        if not trace["evidence"]:
            trace["status"] = "not_found"
            trace["findings"] = []
    return trace


def run_memory_agent(delg: dict, now_epoch: float, con, max_tool_turns: int = 4) -> dict:
    bound = now_epoch
    msgs = [
        {"role": "system", "content": P.MEMORY_AGENT_SYS +
         f'\nSIMULATED NOW: {delg["now_iso"]} (Asia/Shanghai, UTC+08:00).'},
        {"role": "user", "content": _delegation_text(delg)},
    ]
    trace = {"delegation": {k: delg.get(k) for k in
                            ("gaps", "keywords", "entities", "time_hint", "known")},
             "now_iso": delg["now_iso"], "steps": [], "sources": {}, "status": "not_found",
             "findings": [], "evidence": [], "gaps": [], "expandable": [], "note": "",
             "n_tool_turns": 0, "final_ok": False}
    pool: dict = {}
    turns = 0
    while turns < max_tool_turns:
        turns += 1
        p = lm.call("agent-tools", msgs, max_tokens=2200, temperature=0.2, tools=AGENT_TOOLS)
        m = lm.message(p)
        calls = m.get("tool_calls") or []
        if not calls:
            # model chose to stop early; still force the reserved synthesis turn below
            if m.get("content"):
                msgs.append({"role": "assistant", "content": m["content"]})
            break
        msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": calls})
        for c in calls:
            fn = (c.get("function") or {})
            name = fn.get("name")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            res = _dispatch(name, args if isinstance(args, dict) else {}, bound, con)
            _collect_sources(res, pool)
            trace["steps"].append({"turn": turns, "tool": name, "args": args,
                                   "n_results": res.get("returned", res.get("n_items",
                                              len(res.get("results", [])) if res.get("results") else 0)),
                                   "status": res.get("found", True)})
            msgs.append({"role": "tool", "tool_call_id": c.get("id"),
                         "content": json.dumps(res, ensure_ascii=False)})
    trace["n_tool_turns"] = turns
    # RESERVED synthesis turn (tools forbidden) -> guaranteed structured conclusion
    msgs.append({"role": "user", "content": P.AGENT_FINAL})
    p = lm.call("agent-summary", msgs, max_tokens=3000, temperature=0.0, response_json=True)
    final = None
    try:
        final = extract_json(lm.content(p) or "{}")
    except Exception as exc:  # noqa: BLE001
        trace["note"] = f"final parse failed: {type(exc).__name__}"
    if isinstance(final, dict):
        finalize(trace, final, pool, bound, con)
    trace["sources"] = list(pool.keys())
    trace["n_sources"] = len(pool)
    return trace
