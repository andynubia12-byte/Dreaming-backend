"""Main model conditions A / B / C / O.

A = current session only. B = + long-term memory. C = + memory + independent
Memory Agent (structured delegation). O = + memory + ORACLE evidence (diagnostic
only; never produced by the automatic agent, never mixed into C).

The exact payload handed to the main model is recorded per call and is NEVER a
sliced JSON string: we bound item counts and set explicit truncated/has_more flags.
"""
from __future__ import annotations

import json

from . import lm
from . import prompts as P
from . import settings as S
from .agent import run_memory_agent
from .retrieval import query_terms

MAX_EVIDENCE_TEXT = 4000
MAX_EVIDENCE_ITEMS = 14


def _ctx_text(q: dict, limit: int = 10) -> str:
    rows = list(q.get("context", []))[-limit:]
    lines = [f"[{m['role']}] (message_at {m.get('message_at', '?')}) {m['text']}" for m in rows]
    return "\n".join(lines) if lines else "(no earlier messages in this session)"


def _memory_excerpt(memory: str, terms: list[str], max_chars: int = 3500) -> str:
    if not memory:
        return ""
    lines = [ln for ln in memory.splitlines() if ln.strip()]
    scored = []
    for i, ln in enumerate(lines):
        low = ln.lower()
        hits = sum(1 for t in terms if t and t in low)
        if hits:
            scored.append((hits, -i, i))
    scored.sort(reverse=True)
    picked, total = [], 0
    for _, _, i in scored:
        ln = lines[i]
        if total + len(ln) > max_chars:
            continue
        picked.append((i, ln))
        total += len(ln)
        if total >= max_chars * 0.85:
            break
    picked.sort()
    return "\n".join(ln for _, ln in picked)


def _build_payload(trace: dict) -> tuple[dict, str]:
    ev = []
    for e in trace.get("evidence", [])[:MAX_EVIDENCE_ITEMS]:
        text = e.get("text") or ""
        trunc = len(text) > MAX_EVIDENCE_TEXT
        ev.append({"source_id": e.get("source_id"), "message_id": e.get("message_id"),
                   "role": e.get("role"), "message_at": e.get("message_at"),
                   "text": text[:MAX_EVIDENCE_TEXT], "truncated": trunc,
                   "total_chars": len(text)})
    more_ev = max(0, len(trace.get("evidence", [])) - MAX_EVIDENCE_ITEMS)
    payload = {
        "status": trace.get("status"),
        "findings": trace.get("findings", []),
        "evidence": ev,
        "evidence_has_more": more_ev > 0,
        "evidence_more_count": more_ev,
        "gaps": trace.get("gaps", []),
        "expandable_source_ids": trace.get("expandable", []),
        "retrieval_meta": {"n_tool_turns": trace.get("n_tool_turns"),
                           "n_sources_seen": trace.get("n_sources"),
                           "note": trace.get("note", "")},
        "time_note": "message_at = when the message was sent (Asia/Shanghai). "
                     "event_at (if present) = when the described event happened; may be null.",
    }
    return payload, json.dumps(payload, ensure_ascii=False)


def main_system(q: dict, cond: str, memory: str) -> str:
    mem = memory if cond in ("B", "C", "O") else ""
    s = (P.MAIN_SYS
         + f"\n\nSIMULATED NOW: {q.get('now')}  (timezone Asia/Shanghai, UTC+08:00)"
         + "\n\n===== CURRENT SESSION (each message carries message_at) =====\n" + _ctx_text(q))
    if cond in ("B", "C", "O"):
        s += "\n\n===== LONG-TERM MEMORY (compressed; may be stale) =====\n" + (mem or "(empty)")
    if cond == "C":
        s += ("\n\nYou may call `delegate_memory` to ask an independent Memory Agent for past "
              "evidence. Ask for the specific gaps you need; it returns sources with message times.")
    if cond == "O":
        s += ("\n\n===== ORACLE REFERENCE EVIDENCE (diagnostic condition; verified original "
              "messages) =====\n" + json.dumps(q.get("_oracle", []), ensure_ascii=False))
    return s


def _delegation(q: dict, memory: str, args: dict) -> dict:
    terms = query_terms(args.get("gaps") and " ".join(args["gaps"]) or "", q.get("text", ""),
                        *(args.get("keywords") or []), *(args.get("entities") or []), keep_stop=True)
    return {
        "question": q.get("text", ""),
        "now_iso": q.get("now"),
        "session_context": list(q.get("context", []))[-8:],
        "memory_excerpt": _memory_excerpt(memory, terms),
        "gaps": args.get("gaps") or [],
        "keywords": args.get("keywords") or [],
        "entities": args.get("entities") or [],
        "time_hint": args.get("time_hint") or "",
        "known": args.get("known") or "",
    }


def run_condition(q: dict, cond: str, memory: str, con=None, max_delegations: int = 3) -> dict:
    system = main_system(q, cond, memory)
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": q["text"]}]
    rec = {"question_id": q["id"], "cond": cond, "answer": "", "now": q.get("now"),
           "delegations": [], "tool_results": [], "tool_payloads": []}
    if cond != "C":
        p = lm.call("main-" + cond, msgs, max_tokens=4000, temperature=S.ANSWER_TEMPERATURE)
        rec["answer"] = lm.content(p)
        if not rec["answer"].strip():            # reasoning consumed the budget -> retry larger
            p = lm.call("main-" + cond, msgs, max_tokens=12000, temperature=S.ANSWER_TEMPERATURE)
            rec["answer"] = lm.content(p)
        rec["prompt_tokens"] = (p.get("usage") or {}).get("prompt_tokens")
        return rec
    for _ in range(max_delegations):
        p = lm.call("main-c", msgs, max_tokens=4000, temperature=S.ANSWER_TEMPERATURE,
                    tools=[P.DELEGATE_TOOL])
        m = lm.message(p)
        calls = m.get("tool_calls") or []
        if not calls:
            rec["answer"] = m.get("content") or ""
            break
        msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": calls})
        for c in calls:
            fn = c.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            args = args if isinstance(args, dict) else {}
            delg = _delegation(q, memory, args)
            trace = run_memory_agent(delg, S.epoch_of(q["time"]), con)
            payload, payload_str = _build_payload(trace)
            rec["delegations"].append(trace)
            rec["tool_payloads"].append(payload)
            rec["tool_results"].append(payload_str)   # EXACT string the main model received
            msgs.append({"role": "tool", "tool_call_id": c.get("id"), "content": payload_str})
    if not rec["answer"]:
        msgs.append({"role": "user", "content": P.MAIN_FINAL})
        p = lm.call("main-final", msgs, max_tokens=4000, temperature=S.ANSWER_TEMPERATURE)
        rec["answer"] = lm.content(p)
    return rec
