"""Local tools exposed to the Memory Agent: search_history / read_message / expand.

Every result is a fully-valid JSON structure: instead of slicing a JSON string
(the old fixed-character-slice defect) we bound the NUMBER of items and set
explicit `has_more` / `truncated` markers with an exact resume locator.
"""
from __future__ import annotations

import sqlite3

from . import lm
from . import settings as S
from . import store
from .retrieval import phrases, query_terms, rank

MAX_TEXT = 4000          # per-message chars returned by read/expand (explicitly flagged)
DEFAULT_SEARCH_LIMIT = 8
MAX_SEARCH_LIMIT = 20


def _snippet(text: str, qterms: list[str], width: int = 220) -> str:
    low = text.lower()
    pos = min((low.find(t) for t in qterms if t and low.find(t) >= 0), default=0)
    start = max(0, pos - width // 3)
    seg = text[start:start + width]
    prefix = "…" if start > 0 else ""
    suffix = "…" if start + width < len(text) else ""
    return prefix + seg + suffix


def _cap(text: str, n: int = MAX_TEXT) -> tuple[str, bool, int]:
    if len(text) <= n:
        return text, False, len(text)
    return text[:n], True, len(text)


def _msg_at(t: float) -> str:
    return S.now_iso(t)


def search_history(args: dict, bound: float, conv: sqlite3.Connection) -> dict:
    """BM25-ranked, time-guarded search over the pre-`bound` selected branches."""
    query = str(args.get("query") or "").strip()
    kws = args.get("keywords") or []
    ents = args.get("entities") or []
    if isinstance(kws, str):
        kws = [kws]
    if isinstance(ents, str):
        ents = [ents]
    terms = query_terms(query, *kws, *ents, keep_stop=True) or query_terms(query, *kws, *ents)
    qph = phrases(query, *kws, *ents)
    try:
        limit = max(1, min(int(args.get("limit") or DEFAULT_SEARCH_LIMIT), MAX_SEARCH_LIMIT))
    except (TypeError, ValueError):
        limit = DEFAULT_SEARCH_LIMIT
    t_lo = args.get("time_start")
    t_hi = args.get("time_end")
    try:
        lo = S.epoch_of(t_lo) if t_lo else None
    except Exception:  # noqa: BLE001
        lo = None
    try:
        hi = S.epoch_of(t_hi) if t_hi else None
    except Exception:  # noqa: BLE001
        hi = None
    only_conv = args.get("conversation_id")

    cor = store.corpus(conv)
    cand = []
    for i, d in enumerate(cor.docs):
        if not (d["t"] < bound):                       # hard guard (strict)
            continue
        if lo is not None and d["t"] < lo:
            continue
        if hi is not None and d["t"] >= hi:            # never reach/overrun `now`
            continue
        if only_conv and d["conv"] != only_conv:
            continue
        cand.append(i)
    ranked = rank(cor, cand, terms, qph)
    hits, dropped = [], max(0, len(ranked) - limit)
    for idx, score in ranked[:limit]:
        d = cor.docs[idx]
        hits.append({
            "source_id": store.sid(d), "message_id": d["id"], "node_id": d["node"],
            "conversation_id": d["conv"], "role": d["role"],
            "message_at": _msg_at(d["t"]), "time": S.utc_iso(d["t"]),
            "score": round(score, 4), "text_len": len(d["text"]),
            "snippet": _snippet(d["text"], terms),
        })
    lm.local_event("search_history", {"query": query, "n_terms": len(terms), "matched": len(ranked),
                                      "returned": len(hits), "dropped": dropped})
    return {
        "tool": "search_history", "query": query, "terms_used": terms[:24],
        "phrases_used": qph[:12], "bound_message_at": _msg_at(bound), "bound_before_now": True,
        "results": hits, "matched_total": len(ranked), "returned": len(hits),
        "has_more": dropped > 0,
        "more_hint": (f"{dropped} more pre-now matches ranked below the cut; "
                      "raise limit or narrow with time/conversation filters") if dropped else "",
        "note": "Ranked by BM25 + exact phrase/entity boost. Recency is only a tie-break. "
                "All results are strictly before now, selected branch, non-reasoning.",
    }


def read_message(args: dict, bound: float, conv: sqlite3.Connection) -> dict:
    sid = str(args.get("source_id") or args.get("message_id") or "").strip()
    if not sid:
        return {"tool": "read_message", "found": False, "error": "missing source_id"}
    conv_filter = args.get("conversation_id")
    rows = store.find_row(conv, sid, bound, conv_filter)
    lm.local_event("read_message", {"source_id": sid, "rows": len(rows)})
    if not rows:
        return {"tool": "read_message", "source_id": sid, "found": False,
                "error": "not found (or it is at/after now, on a dropped branch, or reasoning-only)"}
    if len(rows) > 1 and not conv_filter:
        return {"tool": "read_message", "source_id": sid, "found": True, "ambiguous": True,
                "candidates": [{"source_id": store.sid(r), "conversation_id": r["conv"],
                                "message_at": _msg_at(r["t"]), "role": r["role"]} for r in rows],
                "hint": "same message_id in multiple conversations; pass conversation_id to disambiguate"}
    r = rows[0]
    text, trunc, total = _cap(r["text"], int(args.get("max_chars") or MAX_TEXT))
    return {"tool": "read_message", "source_id": store.sid(r), "found": True, "ambiguous": False,
            "item": {"source_id": store.sid(r), "message_id": r["id"], "node_id": r["node"],
                     "conversation_id": r["conv"], "role": r["role"],
                     "message_at": _msg_at(r["t"]), "time": S.utc_iso(r["t"]),
                     "text": text, "truncated": trunc, "total_chars": total,
                     "resume_hint": (f"truncated at {len(text)} chars; full text has {total} chars"
                                     if trunc else "")}}


def expand(args: dict, bound: float, conv: sqlite3.Connection) -> dict:
    """Neighbours in the SAME conversation's selected-branch order, target included.

    Fixes the old 'start 3000s before target, take earliest 5' bug: we locate the
    target by its real order index and take a symmetric window around it.
    """
    sid = str(args.get("source_id") or args.get("message_id") or "").strip()
    rows = store.find_row(conv, sid, bound, args.get("conversation_id")) if sid else []
    lm.local_event("expand", {"source_id": sid, "rows": len(rows)})
    if not rows:
        return {"tool": "expand", "source_id": sid, "found": False,
                "error": "target not found before now (or dropped branch/reasoning)"}
    if len(rows) > 1 and not args.get("conversation_id"):
        return {"tool": "expand", "source_id": sid, "found": True, "ambiguous": True,
                "candidates": [{"source_id": store.sid(r), "conversation_id": r["conv"],
                                "message_at": _msg_at(r["t"])} for r in rows],
                "hint": "ambiguous id; pass conversation_id"}
    target = rows[0]
    try:
        before = max(0, min(int(args.get("before", 2)), 12))
        after = max(0, min(int(args.get("after", 2)), 12))
        offset = int(args.get("offset", 0))
    except (TypeError, ValueError):
        before, after, offset = 2, 2, 0
    thread = store.conversation_thread(conv, target["conv"], bound)
    idx = next((i for i, d in enumerate(thread)
                if d["id"] == target["id"] and d["node"] == target["node"]), None)
    if idx is None:
        return {"tool": "expand", "source_id": sid, "found": False,
                "error": "target vanished from its own thread (unexpected)"}
    lo = max(0, idx - before + offset)
    hi = min(len(thread), idx + after + 1 + offset)
    if lo > idx or hi <= idx:
        lo, hi = max(0, idx - before), min(len(thread), idx + after + 1)  # never lose the target
    items = []
    for j in range(lo, hi):
        d = thread[j]
        text, trunc, total = _cap(d["text"])
        items.append({"source_id": store.sid(d), "message_id": d["id"], "node_id": d["node"],
                      "conversation_id": d["conv"], "role": d["role"],
                      "message_at": _msg_at(d["t"]), "time": S.utc_iso(d["t"]),
                      "text": text, "truncated": trunc, "total_chars": total,
                      "is_target": (j == idx)})
    return {"tool": "expand", "found": True, "ambiguous": False, "target": store.sid(target),
            "target_index": idx, "thread_len": len(thread),
            "window": [lo, hi - 1], "items": items, "n_items": len(items),
            "has_more_before": lo > 0, "has_more_after": hi < len(thread),
            "resume": {"before_next_offset": offset - max(0, idx - before) + lo if lo > 0 else None,
                       "after_next_offset": offset + (hi - 1) - idx if hi < len(thread) else None},
            "note": "Same-conversation selected-branch order; the target is always included."}
