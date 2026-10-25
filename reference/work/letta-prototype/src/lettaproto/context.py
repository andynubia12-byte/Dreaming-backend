"""Context assembly: inject the FULL working idea.md (no compression into a profile).

Token figures are clearly labelled ESTIMATES; we do not claim an estimate equals a
verified model capacity. Truncation is reported explicitly, never silent.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

from . import ideastore
from .config import Config
from .util import approx_tokens, iso_from_epoch

PREAMBLE = (
    "You are a personal assistant with a long-term memory document about the user, "
    "reproduced below in full. Treat it as background context the user has curated. "
    "Reply in the user's language, warmly and specifically, and prefer the most recent "
    "explicitly dated statements when the document disagrees with itself. "
    "Do not treat assistant-written analysis as a user fact."
)


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
            "\n===== WORKING MEMORY (idea.md, full) =====\n",
            self.memory_text,
        ]
        if recent:
            parts.append("\n===== RECENT EXCHANGES (oldest to newest) =====\n" + recent)
        return "".join(parts)


def recent_from_history(
    conn: sqlite3.Connection, *, limit: int, token_budget: int
) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT conversation_id, message_id, role, create_time, text FROM messages "
        "WHERE role IN ('user','assistant') AND is_assistant_reasoning=0 AND is_selected_branch=1 "
        "ORDER BY create_time DESC LIMIT ?",
        (limit,),
    ).fetchall()
    kept: list[dict[str, Any]] = []
    used = 0
    for row in rows:
        t = approx_tokens(row["text"] or "")
        if kept and used + t > token_budget:
            break
        used += t
        kept.append(
            {
                "conversation_id": row["conversation_id"],
                "message_id": row["message_id"],
                "role": row["role"],
                "time": iso_from_epoch(row["create_time"]),
                "text": row["text"],
            }
        )
    return list(reversed(kept))


def recent_from_chatlog(
    conn: sqlite3.Connection, instance: str, *, limit: int, token_budget: int | None = None, conversation_id: str | None = None
) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT conversation_id, role, created_at, content FROM chat_log WHERE instance=? "
        "AND (? IS NULL OR conversation_id=?) ORDER BY id DESC LIMIT ?",
        (instance, conversation_id, conversation_id, limit),
    ).fetchall()
    kept: list[dict[str, Any]] = []
    used = 0
    budget = token_budget if token_budget and token_budget > 0 else None
    for r in rows:  # newest first
        t = approx_tokens(r["content"] or "")
        if budget is not None and kept and used + t > budget:
            break
        used += t
        kept.append(
            {"conversation_id": r["conversation_id"], "role": r["role"], "time": iso_from_epoch(r["created_at"]), "text": r["content"]}
        )
    return list(reversed(kept))


def build_context(
    config: Config,
    conn: sqlite3.Connection,
    *,
    instance: str = "prod",
    recent_limit: int | None = None,
    recent_token_budget: int | None = None,
    conversation_id: str | None = None,
    model_context_tokens: int = 1_000_000,
    from_chatlog: bool = True,
    now: str | None = None,
) -> ContextBundle:
    if recent_limit is None:
        recent_limit = config.default_recent_messages
    if recent_token_budget is None:
        recent_token_budget = config.default_recent_token_budget
    memory = ideastore.load(config, instance)
    mem_tokens = approx_tokens(memory)
    recent: list[dict[str, Any]] = []
    if recent_limit > 0:
        if from_chatlog:
            recent = recent_from_chatlog(conn, instance, limit=recent_limit, token_budget=recent_token_budget, conversation_id=conversation_id)
        else:
            recent = recent_from_history(conn, limit=recent_limit, token_budget=recent_token_budget)

    system_prompt_chars = len(PREAMBLE) + len(memory) + sum(len(m["text"] or "") for m in recent)
    est = approx_tokens(PREAMBLE) + mem_tokens + sum(approx_tokens(m["text"] or "") for m in recent)
    bundle = ContextBundle(memory_text=memory, memory_instance=instance, recent=recent, now=now)
    bundle.stats = {
        "instance": instance,
        "now": now,
        "now_injected": bool(now),
        "recent_source": "chat_log" if from_chatlog else "export_history",
        "memory_chars": len(memory),
        "memory_token_estimate": mem_tokens,
        "memory_injected_full": True,
        "memory_truncated": False,
        "recent_messages": len(recent),
        "recent_chars": sum(len(m["text"] or "") for m in recent),
        "recent_token_estimate": sum(approx_tokens(m["text"] or "") for m in recent),
        "system_prompt_chars": system_prompt_chars,
        "system_prompt_token_estimate": est,
        "model_context_tokens_declared": model_context_tokens,
        "estimate_note": "token counts are ESTIMATES, not measured model capacity",
    }
    return bundle
