"""Context assembly for the continuity trial.

Adapted from ``work/generated-memory-demo/src/lettaproto/context.py``
(sha256 07d4e4434a14d76de9bb09ec47f9f1c92e7955f097abc5abac51633ea4ba0173).

Differences:
* Working memory is the FULL *published* summary from the maintenance engine
  (via ``summary_store``), never the production ``idea.md``/``stable.md`` and never
  a half-written version. The former two-layer assembly is removed.
* The preamble now marks the summary as an auto-maintained, correctable system
  memory, not a user-curated document.
* Recent chat is filtered to the current ``conversation_id`` (per-session), while
  the summary is shared across all sessions (single-user long memory).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

from .summary_store import Published
from .util import approx_tokens, iso_from_epoch, looks_like_tool_serialization

PREAMBLE = (
    "You are a personal assistant with a long-term memory summary about the user, "
    "reproduced below in full. It is an automatically maintained, evidence-backed "
    "system memory that can be and often is corrected over time; it is NOT endorsed "
    "line-by-line by the user. Treat it as background understanding, prefer the most "
    "recent explicitly dated statements when it disagrees with itself, and never treat "
    "assistant-written analysis as a user fact. If the summary says no memory exists "
    "yet, do not pretend to know the user's personal history; rely only on this session."
)

SUMMARY_HEADER = "# 常驻摘要（自动维护，可修正）"


def assemble_memory(published: Published) -> dict[str, Any]:
    """Return the exact text injected as shared long-term memory."""
    text = published.display_text
    return {
        "text": text,
        "present": published.present,
        "version": published.version,
        "published_at": published.published_at,
        "summary_path": published.summary_path,
        "reason": published.reason,
        "assembly": "published_summary(full)" if published.present else "empty_memory",
    }


@dataclass
class ContextBundle:
    memory_text: str
    memory_instance: str
    recent: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    now: str | None = None

    def system_prompt(self) -> str:
        recent = "\n".join(
            f"[{m['role']}] (message_at {m.get('time') or '?'}) {m['text']}" for m in self.recent
        )
        parts = [
            PREAMBLE,
            (f"\n\nSIMULATED NOW: {self.now}  (timezone Asia/Shanghai, UTC+08:00). "
             "Only history before this moment exists; resolve relative time words against each "
             "message's own message_at, and never present a plan as if it already happened."
             if self.now else ""),
            f"\n\n===== {SUMMARY_HEADER} =====\n",
            self.memory_text,
        ]
        if recent:
            parts.append("\n===== RECENT EXCHANGES (this session only, oldest to newest) =====\n" + recent)
        return "".join(parts)


def recent_from_chatlog(
    conn: sqlite3.Connection,
    instance: str,
    *,
    limit: int,
    token_budget: int | None = None,
    conversation_id: str | None = None,
) -> list[dict[str, Any]]:
    """Recent turns for ONE session only (session isolation), oldest->newest.

    Failed/leaked assistant turns (empty or raw tool serialization) are never
    replayed into context; the underlying row is left untouched.
    """
    rows = conn.execute(
        "SELECT conversation_id, role, created_at, content FROM chat_log WHERE instance=? "
        "AND (? IS NULL OR conversation_id=?) ORDER BY id DESC LIMIT ?",
        (instance, conversation_id, conversation_id, limit),
    ).fetchall()
    kept: list[dict[str, Any]] = []
    used = 0
    budget = token_budget if token_budget and token_budget > 0 else None
    for r in rows:  # newest first
        text = r["content"] or ""
        if r["role"] == "assistant" and (not text.strip() or looks_like_tool_serialization(text)):
            continue
        t = approx_tokens(text)
        if budget is not None and kept and used + t > budget:
            break
        used += t
        kept.append(
            {"conversation_id": r["conversation_id"], "role": r["role"],
             "time": iso_from_epoch(r["created_at"]), "text": text}
        )
    return list(reversed(kept))


def build_context(
    published: Published,
    conn: sqlite3.Connection,
    *,
    instance: str = "trial",
    recent_limit: int = 40,
    recent_token_budget: int = 60_000,
    now: str | None = None,
    conversation_id: str | None = None,
    model_context_tokens: int = 1_000_000,
) -> ContextBundle:
    assembled = assemble_memory(published)
    memory = assembled["text"]
    recent = (
        recent_from_chatlog(conn, instance, limit=recent_limit,
                            token_budget=recent_token_budget, conversation_id=conversation_id)
        if recent_limit and recent_limit > 0 else []
    )
    bundle = ContextBundle(memory_text=memory, memory_instance=instance, recent=recent, now=now)
    bundle.stats = {
        "instance": instance,
        "now": now,
        "now_injected": bool(now),
        "recent_source": "chat_log",
        "conversation_id": conversation_id,
        "memory_present": assembled["present"],
        "memory_version": assembled["version"],
        "memory_assembly": assembled["assembly"],
        "memory_chars": len(memory),
        "memory_token_estimate": approx_tokens(memory),
        "memory_injected_full": True,
        "memory_truncated": False,
        "recent_messages": len(recent),
        "recent_chars": sum(len(m["text"] or "") for m in recent),
        "recent_token_estimate": sum(approx_tokens(m["text"] or "") for m in recent),
        "system_prompt_chars": len(PREAMBLE) + len(memory) + sum(len(m["text"] or "") for m in recent),
        "model_context_tokens_declared": model_context_tokens,
        "estimate_note": "token counts are ESTIMATES, not measured model capacity",
    }
    return bundle
