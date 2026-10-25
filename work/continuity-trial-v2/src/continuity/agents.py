"""Independent Memory Agent search entry (Flash role).

Adapted from ``work/generated-memory-demo/src/lettaproto/agents.py``
(sha256 b3c94848d832b21cdec8ac9e9678ebc067cf6e505d08751886652dd5feb1efd3),
reduced to the read-only search job:

* the *retrieval* agent (``memory_agent.py``) is reused verbatim and searches this
  trial's OWN chat store only (``retrieval.py`` also reads the ``messages`` table,
  which this trial keeps empty — no history is imported);
* the memory excerpt comes from the *published summary*, not the production
  ``idea.md``/``stable.md``.

The write-side consolidate path (and the search-trace table) is intentionally not
copied: this trial never has the main chat write memory directly.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from . import memory_agent
from .config import Config
from .deepseek import DeepSeekClient
from .retrieval import excerpt, now_shanghai
from .summary_store import Published
from .util import looks_like_tool_serialization

AGENT_VERSION = memory_agent.AGENT_VERSION


def _session_context(
    conn: sqlite3.Connection, instance: str, exclude_chat_ids: tuple, *,
    limit: int = 8, conversation_id: str | None = None,
) -> list[dict]:
    """Recent turns of THIS session as bounded context; current turn excluded."""
    exclude = {str(x) for x in exclude_chat_ids}
    rows = conn.execute(
        "SELECT id, created_at, role, content FROM chat_log WHERE instance=? "
        "AND role IN ('user','assistant') AND (? IS NULL OR conversation_id=?) ORDER BY id DESC LIMIT ?",
        (instance, conversation_id, conversation_id, limit + len(exclude)),
    ).fetchall()
    out: list[dict] = []
    for r in reversed(rows):
        if str(r["id"]) in exclude:
            continue
        text = r["content"] or ""
        if r["role"] == "assistant" and (not text.strip() or looks_like_tool_serialization(text)):
            continue
        out.append({"role": r["role"], "message_at": now_shanghai(float(r["created_at"])), "text": text})
    return out[-limit:]


def _build_delegation(query_text: str, delegation: dict | None) -> dict:
    d = dict(delegation or {})
    gaps = d.get("gaps") or ([query_text] if query_text else [])
    d["gaps"] = list(gaps) if isinstance(gaps, (list, tuple)) else [str(gaps)]
    d.setdefault("question", query_text or (d["gaps"][0] if d["gaps"] else ""))
    d.setdefault("keywords", [])
    d.setdefault("entities", [])
    d.setdefault("time_hint", "")
    d.setdefault("known", "")
    return d


def memory_agent_search(
    config: Config,
    conn: sqlite3.Connection,
    published: Published,
    query_text: str,
    *,
    instance: str = "trial",
    client: DeepSeekClient | None = None,
    max_rounds: int | None = None,
    now_epoch: float | None = None,
    delegation: dict | None = None,
    exclude_chat_ids: tuple = (),
    conversation_id: str | None = None,
) -> dict[str, Any]:
    """Delegate a structured lookup to the independent Memory Agent (Flash)."""
    client = client or DeepSeekClient(config)
    now = float(now_epoch) if now_epoch is not None else __import__("time").time()
    delg = _build_delegation(query_text, delegation)
    delg["now_iso"] = now_shanghai(now)
    delg["session_context"] = _session_context(
        conn, instance, exclude_chat_ids, conversation_id=conversation_id
    )
    delg["memory_excerpt"] = excerpt(
        published.display_text,
        " ".join(
            [str(delg.get("question") or "")]
            + [str(g) for g in delg["gaps"]]
            + [str(k) for k in delg.get("keywords") or []]
            + [str(e) for e in delg.get("entities") or []]
        ),
    )
    result = memory_agent.run(
        config, conn, instance, delg, now, client=client,
        max_tool_turns=int(max_rounds) if max_rounds else 4,
        exclude_chat_ids=tuple(exclude_chat_ids),
    )
    status = result.get("status", "not_found")
    return {
        **result,
        "query": query_text or delg.get("question", ""),
        "queries": [query_text] if query_text else [delg.get("question", "")],
        "sufficient": status in ("found", "partial"),
        "rounds": result.get("n_tool_turns", 0),
    }
