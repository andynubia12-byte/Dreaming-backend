"""Read-only access to the prototype canonical DB + a cached BM25 corpus.

Hard guards live here and are enforced in SQL AND in Python:
  * create_time < bound   (the question's own simulated now, never the run day)
  * is_selected_branch = 1 (abandoned branches are never returned)
  * is_assistant_reasoning = 0 (private reasoning is never returned)
Correctly parenthesised OR groups are used so the guards apply to EVERY term
(the old code's `A AND B AND c OR d` re-opened the filter after the first OR).
"""
from __future__ import annotations

import sqlite3

from . import settings as S
from .retrieval import Corpus

# Corpus cache keyed by the underlying DB file, so a synthetic fixture and the
# real canonical DB can be used in the same process without cross-contamination.
_CORPUS: dict[str, Corpus] = {}


def ro() -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{S.DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only=1")
    return con


def load_docs(con: sqlite3.Connection | None = None) -> list[dict]:
    """Selected-branch, non-reasoning, non-empty messages (the searchable corpus)."""
    opened = con is None
    con = con or ro()
    try:
        rows = con.execute(
            "SELECT message_id, node_id, conversation_id, role, create_time, text "
            "FROM messages WHERE is_selected_branch=1 AND is_assistant_reasoning=0 "
            "AND text IS NOT NULL AND LENGTH(text)>0 ORDER BY create_time, node_id"
        ).fetchall()
    finally:
        if opened:
            con.close()
    return [{"id": r["message_id"], "node": r["node_id"], "conv": r["conversation_id"],
             "role": r["role"], "t": float(r["create_time"]), "text": r["text"]} for r in rows]


def db_key(con: sqlite3.Connection) -> str:
    try:
        rows = con.execute("PRAGMA database_list").fetchall()
        return str(rows[0][2]) if rows else "?"
    except Exception:  # noqa: BLE001
        return "?"


def corpus(con: sqlite3.Connection | None = None, refresh: bool = False) -> Corpus:
    opened = con is None
    con = con or ro()
    try:
        key = db_key(con)
        if refresh or key not in _CORPUS:
            docs = load_docs(con)
            seen, uniq = set(), []
            for d in docs:
                k = (d["id"], d["conv"])
                if k in seen:
                    continue
                seen.add(k)
                uniq.append(d)
            _CORPUS[key] = Corpus(uniq)
        return _CORPUS[key]
    finally:
        if opened:
            con.close()


def guarded_rows(con: sqlite3.Connection, terms: list[str], bound: float, limit: int) -> list[sqlite3.Row]:
    """SQL prefilter that DEMONSTRABLY applies every guard to every OR term."""
    if not terms:
        return []
    like = " OR ".join("text LIKE ?" for _ in terms)
    sql = (
        "SELECT message_id, node_id, conversation_id, role, create_time, text FROM messages "
        "WHERE is_selected_branch=1 AND is_assistant_reasoning=0 "
        "AND create_time < ? AND ( " + like + " ) "
        "ORDER BY create_time DESC LIMIT ?"
    )
    return con.execute(sql, [bound, *[f"%{t}%" for t in terms], limit]).fetchall()


def conversation_thread(con: sqlite3.Connection, conv_id: str, bound: float) -> list[dict]:
    """Selected-branch readable thread (non-empty text), ordered deterministically.

    Ordering is (create_time, node_id) so same-second messages and duplicate ids
    stay stable and unambiguous.
    """
    rows = con.execute(
        "SELECT message_id, node_id, conversation_id, role, create_time, text FROM messages "
        "WHERE conversation_id=? AND is_selected_branch=1 AND is_assistant_reasoning=0 "
        "AND text IS NOT NULL AND LENGTH(text)>0 AND create_time < ? "
        "ORDER BY create_time, node_id",
        (conv_id, bound),
    ).fetchall()
    return [{"id": r["message_id"], "node": r["node_id"], "conv": r["conversation_id"],
             "role": r["role"], "t": float(r["create_time"]), "text": r["text"]} for r in rows]


def find_row(con: sqlite3.Connection, source_id: str, bound: float,
             conv_id: str | None = None) -> list[dict]:
    """Resolve a source_id (message_id or `conv#node`) to guarded rows.

    Returns a LIST: >1 means the id is genuinely ambiguous (same message_id in
    two conversations) and the caller/agent must disambiguate with conv_id.
    """
    if "#" in source_id:
        conv_id, node = source_id.split("#", 1)
        rows = con.execute(
            "SELECT message_id, node_id, conversation_id, role, create_time, text FROM messages "
            "WHERE conversation_id=? AND node_id=? AND is_selected_branch=1 "
            "AND is_assistant_reasoning=0 AND create_time < ?",
            (conv_id, node, bound),
        ).fetchall()
        if not rows:
            # be lenient: the caller may have mis-typed conv#message_id; retry on either part
            for cand in (node, source_id):
                rows = con.execute(
                    "SELECT message_id, node_id, conversation_id, role, create_time, text FROM messages "
                    "WHERE message_id LIKE ? AND is_selected_branch=1 AND is_assistant_reasoning=0 "
                    "AND create_time < ? LIMIT 5",
                    (cand[:36] + "%", bound),
                ).fetchall()
                if rows:
                    break
    else:
        def _q(op: str, mid: str, cid: str | None):
            sql = ("SELECT message_id, node_id, conversation_id, role, create_time, text FROM messages "
                   f"WHERE message_id{op}? AND is_selected_branch=1 AND is_assistant_reasoning=0 "
                   "AND create_time < ?")
            a: list = [mid, bound]
            if cid:
                sql += " AND conversation_id=?"
                a.append(cid)
            return con.execute(sql, a).fetchall()
        rows = _q("=", source_id, conv_id)
        if not rows and len(source_id) < 36:      # allow a unique message_id prefix
            rows = _q(" LIKE ", source_id + "%", conv_id)
    return [{"id": r["message_id"], "node": r["node_id"], "conv": r["conversation_id"],
             "role": r["role"], "t": float(r["create_time"]), "text": r["text"]} for r in rows]


def sid(doc: dict) -> str:
    """Stable, unambiguous source id: `<conversation_id>#<node_id>` (the table PK).

    message_id alone is ambiguous (19 ids appear in two conversations), and same
    second timestamps exist, so the (conv,node) pair is the reliable locator.
    """
    return f'{doc["conv"]}#{doc["node"]}'
