"""Incremental updater for the working idea.md.

Only NEW exchanges (prototype chat) or an explicit user correction are processed.
History is never re-mined. Updates APPEND a dated, sourced section; the original
document text is preserved (append/correction, never a full rewrite). Every applied
update is versioned and rollback-able; a failed model call leaves the copy unchanged.

No silent truncation: the full working memory and the full new messages are sent.
If the assembled input exceeds the configured ceiling, we FAIL loudly and change
nothing (neither the memory nor the cursor).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from . import db, ideastore
from .config import Config
from .deepseek import BudgetExceeded, DeepSeekClient
from .util import approx_tokens

UPDATE_SYSTEM = """You maintain an incremental log for a user's long-term memory document.

You are given (a) the CURRENT memory document text and (b) ONLY the new user messages, or an
explicit correction from the user. Produce a SINGLE new Markdown section that records what is
new or corrected. Rules:
- Do NOT rewrite or repeat the whole document; output only the new section to APPEND.
- Start the section with a dated heading: `### 更新 YYYY-MM-DD`.
- Be precise and specific; keep provenance by referencing the given message ids in parentheses.
- If a correction supersedes an earlier statement, say what it replaces.
- Never invent facts. If nothing durable is present, return {"section_markdown": ""}.
- Output JSON: {"section_markdown": "...", "summary": "one line", "sources": ["id", ...]}."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cursor_key(instance: str) -> str:
    return f"update_cursor:{instance}"


def append_update(
    config: Config,
    conn: sqlite3.Connection,
    instance: str,
    section_md: str,
    *,
    note: str,
    sources: list[Any],
    summary: str,
    trigger: str,
    prev_version: int,
) -> dict[str, Any]:
    current = ideastore.load(config, instance)
    new_text = current.rstrip() + "\n\n" + section_md.strip() + "\n"
    stat = ideastore.diff_stat(current, new_text)
    info = ideastore.save(config, conn, new_text, note=note, instance=instance)
    conn.execute(
        "INSERT INTO memory_updates(created_at, instance, trigger, prev_version, new_version, diff_added, diff_removed, sources_json, summary, status)"
        " VALUES(?,?,?,?,?,?,?,?,?,?)",
        (_now(), instance, trigger, prev_version, info["version"], stat["added"], stat["removed"],
         json.dumps(sources, ensure_ascii=False), summary, "ok"),
    )
    conn.commit()
    info["diff_stat"] = stat
    return info


def _chatlog_rows(conn: sqlite3.Connection, instance: str, cursor: float) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT id, created_at, role, content FROM chat_log WHERE instance=? AND role='user' AND created_at > ? ORDER BY created_at",
        (instance, cursor),
    ).fetchall()


def build_input(config: Config, current: str, rows: list[sqlite3.Row]) -> tuple[str, int]:
    """Assemble the FULL updater input; return (text, estimated tokens). No truncation."""
    lines = ["CURRENT MEMORY DOCUMENT (full):\n" + current, "\nNEW USER MESSAGES:"]
    for row in rows:
        lines.append(json.dumps({"id": f"chat#{row['id']}", "time": row["created_at"], "text": row["content"]}, ensure_ascii=False))
    text = "\n".join(lines)
    return text, approx_tokens(text)


def update_from_new_messages(
    config: Config,
    conn: sqlite3.Connection,
    client: DeepSeekClient,
    instance: str = "prod",
    *,
    apply: bool = True,
    max_messages: int = 200,
) -> dict[str, Any]:
    cursor = float(db.get_meta(conn, _cursor_key(instance), "0") or 0)
    rows = _chatlog_rows(conn, instance, cursor)[:max_messages]
    if not rows:
        return {"status": "noop", "instance": instance, "new_messages": 0, "cursor_advanced": False}

    current = ideastore.load(config, instance)  # full, never sliced
    prompt, est_tokens = build_input(config, current, rows)
    if est_tokens > config.updater_max_input_tokens:
        # explicit failure; nothing is dropped and nothing is written
        raise BudgetExceeded(
            f"updater input estimated {est_tokens} tokens exceeds ceiling "
            f"{config.updater_max_input_tokens}; refusing to truncate. Memory and cursor unchanged."
        )

    result = client.ask_json(UPDATE_SYSTEM, prompt, max_tokens=4000)
    section = (result.get("section_markdown") or "").strip()
    sources = result.get("sources") or [f"chat#{r['id']}" for r in rows]
    max_ts = max(r["created_at"] for r in rows)

    if not section:
        if apply:
            db.set_meta(conn, _cursor_key(instance), str(max_ts))
            conn.commit()
        return {
            "status": "noop",
            "reason": "model found nothing durable",
            "new_messages": len(rows),
            "input_token_estimate": est_tokens,
            "cursor_advanced": bool(apply),
        }
    if not apply:
        return {
            "status": "dry-run",
            "section_markdown": section,
            "sources": sources,
            "new_messages": len(rows),
            "input_token_estimate": est_tokens,
            "cursor_advanced": False,
        }

    prev = ideastore.current_version(conn, instance)
    info = append_update(
        config, conn, instance, section,
        note=f"dreaming update from {len(rows)} new messages",
        sources=sources, summary=result.get("summary") or "", trigger="new_messages", prev_version=prev,
    )
    db.set_meta(conn, _cursor_key(instance), str(max_ts))
    conn.commit()
    return {
        "status": "ok",
        "new_messages": len(rows),
        "version": info["version"],
        "diff_stat": info["diff_stat"],
        "sources": sources,
        "summary": result.get("summary") or "",
        "input_token_estimate": est_tokens,
        "cursor_advanced": True,
    }


def update_from_correction(
    config: Config,
    conn: sqlite3.Connection,
    client: DeepSeekClient,
    correction: str,
    instance: str = "prod",
    *,
    apply: bool = True,
) -> dict[str, Any]:
    current = ideastore.load(config, instance)  # full, never sliced
    user = "CURRENT MEMORY DOCUMENT (full):\n" + current + "\n\nEXPLICIT USER CORRECTION (verbatim, untruncated):\n" + correction
    est_tokens = approx_tokens(user)
    if est_tokens > config.updater_max_input_tokens:
        raise BudgetExceeded(
            f"correction input estimated {est_tokens} tokens exceeds ceiling "
            f"{config.updater_max_input_tokens}; refusing to truncate. Memory unchanged."
        )
    result = client.ask_json(UPDATE_SYSTEM, user, max_tokens=4000)
    section = (result.get("section_markdown") or "").strip()
    if not section:
        return {"status": "noop", "reason": "model found nothing to record", "cursor_advanced": False}
    if not apply:
        return {"status": "dry-run", "section_markdown": section, "sources": result.get("sources") or [], "cursor_advanced": False}
    prev = ideastore.current_version(conn, instance)
    info = append_update(
        config, conn, instance, section,
        note="explicit user correction", sources=result.get("sources") or ["user-correction"],
        summary=result.get("summary") or "", trigger="correction", prev_version=prev,
    )
    return {"status": "ok", "version": info["version"], "diff_stat": info["diff_stat"], "section_markdown": section}
