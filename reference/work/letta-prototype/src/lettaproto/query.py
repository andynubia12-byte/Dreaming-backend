"""On-demand query: prefer the working idea.md; fall back to local history.

History hits always carry time / role / conversation_id / node_id / message_id /
selected-branch flag so an answer can be traced to a source.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from . import ideastore
from .config import Config
from .util import iso_from_epoch

CJK = r"\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_WORD_RE = re.compile(rf"[A-Za-z0-9_]{{2,}}|[{CJK}]+")


def tokenize(text: str) -> list[str]:
    terms: list[str] = []
    for match in _WORD_RE.finditer(text or ""):
        token = match.group(0)
        if re.fullmatch(rf"[{CJK}]+", token):
            if len(token) == 1:
                terms.append(token)
            else:
                terms.extend(token[i : i + 2] for i in range(len(token) - 1))
        else:
            terms.append(token.lower())
    # de-dup, keep order
    seen: set[str] = set()
    out: list[str] = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


@dataclass
class Section:
    heading: str
    start_line: int
    end_line: int
    content: str


def split_sections(text: str) -> list[Section]:
    lines = text.splitlines()
    sections: list[Section] = []
    cur_head = "(preamble)"
    cur_start = 1
    buf: list[str] = []
    for idx, line in enumerate(lines, start=1):
        if re.match(r"^#{1,6}\s+", line):
            if buf or sections:
                sections.append(Section(cur_head, cur_start, idx - 1, "\n".join(buf)))
            cur_head, cur_start, buf = line.strip(), idx, [line]
        else:
            buf.append(line)
    if buf:
        sections.append(Section(cur_head, cur_start, len(lines), "\n".join(buf)))
    return sections


def _score(terms: list[str], text: str) -> float:
    low = text.lower()
    return float(sum(low.count(t) for t in terms))


def search_idea(config: Config, instance: str, question: str, top_k: int = 5) -> list[dict[str, Any]]:
    try:
        text = ideastore.load(config, instance)
    except FileNotFoundError:
        return []
    terms = tokenize(question)
    scored = []
    for sec in split_sections(text):
        s = _score(terms, sec.content)
        if s > 0:
            scored.append((s, sec))
    scored.sort(key=lambda x: (-x[0], x[1].start_line))
    out = []
    for s, sec in scored[:top_k]:
        out.append(
            {
                "source": "idea.md",
                "instance": instance,
                "heading": sec.heading.lstrip("# ").strip(),
                "lines": f"{sec.start_line}-{sec.end_line}",
                "score": s,
                "excerpt": sec.content[:600],
            }
        )
    return out


def search_history(
    conn: sqlite3.Connection,
    question: str,
    *,
    top_k: int = 8,
    selected_only: bool = True,
) -> list[dict[str, Any]]:
    terms = [t for t in tokenize(question) if len(t) >= 2][:6]
    if not terms:
        return []
    where = " AND ".join("m.text LIKE ?" for _ in terms)
    args: list[Any] = [f"%{t}%" for t in terms]
    sql = (
        "SELECT m.conversation_id, m.node_id, m.message_id, m.role, m.create_time, "
        "m.is_selected_branch, m.text, c.title "
        "FROM messages m LEFT JOIN conversations c ON c.id=m.conversation_id "
        f"WHERE m.is_assistant_reasoning=0 AND {where}"
    )
    if selected_only:
        sql += " AND m.is_selected_branch=1"
    sql += " ORDER BY m.create_time DESC LIMIT ?"
    rows = conn.execute(sql, [*args, top_k]).fetchall()
    out = []
    for row in rows:
        text = row["text"] or ""
        idx = min((text.find(t) for t in terms if text.find(t) >= 0), default=0)
        out.append(
            {
                "source": "history",
                "conversation_id": row["conversation_id"],
                "node_id": row["node_id"],
                "message_id": row["message_id"],
                "role": row["role"],
                "time": iso_from_epoch(row["create_time"]),
                "selected_branch": bool(row["is_selected_branch"]),
                "title": row["title"],
                "excerpt": text[max(0, idx - 60) : idx + 240],
            }
        )
    return out


def query(
    config: Config,
    conn: sqlite3.Connection,
    question: str,
    *,
    instance: str = "prod",
    top_k: int = 5,
    history_top_k: int = 8,
    include_history: bool = False,
) -> dict[str, Any]:
    idea_hits = search_idea(config, instance, question, top_k=top_k)
    used_history = include_history or not idea_hits
    history_hits = search_history(conn, question, top_k=history_top_k) if used_history else []
    return {
        "question": question,
        "instance": instance,
        "idea_hits": idea_hits,
        "history_hits": history_hits,
        "strategy": "idea+history" if (idea_hits and history_hits) else ("idea" if idea_hits else "history"),
    }
