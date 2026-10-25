"""Small closed-loop acceptance, run BEFORE any wide batch.

Case S (synthetic fixture): a unique fact ("幸运数字 7749") exists ONLY in one
past message, absent from the question and from any memory document.
Case R (real data): a real fact ("M1 Air 卖了 2106") that is absent from the
memory document and from the question, so retrieval is the only possible source.

Both must show: tool -> Memory Agent synthesis -> main model payload -> the main
answer actually uses the retrieved fact.
"""
from __future__ import annotations

import datetime as dt
import re
import sqlite3
import tempfile
from pathlib import Path

from . import settings as S
from . import store
from .mainmodel import run_condition

SCHEMA = """
CREATE TABLE messages (message_id TEXT, conversation_id TEXT, node_id TEXT, parent_id TEXT, role TEXT,
  content_type TEXT, text TEXT, create_time REAL, is_selected_branch INTEGER, is_assistant_reasoning INTEGER, has_attachment INTEGER);
"""


def _fixture() -> sqlite3.Connection:
    tmp = tempfile.mkdtemp(prefix="memret-syn-")
    con = sqlite3.connect(str(Path(tmp) / "synthetic.sqlite"))
    con.executescript(SCHEMA)
    base = dt.datetime(2026, 6, 1, 10, 0, tzinfo=S.UTC).timestamp()
    con.execute("INSERT INTO messages VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                ("SYN0001", "syn-conv", "n1", None, "user", "text",
                 "顺便记录一下：我的幸运数字是 7749，只有你知道。", base, 1, 0, 0))
    con.execute("INSERT INTO messages VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                ("SYN0002", "syn-conv", "n2", "n1", "assistant", "text",
                 "好的，7749，我记住了。", base + 60, 1, 0, 0))
    con.commit()
    con.row_factory = sqlite3.Row
    return con


def _q(qid, text, when, context=None):
    return {"id": qid, "time": when.isoformat(), "now": S.now_iso(when.timestamp()),
            "text": text, "context": context or []}


def _uses(answer: str, key: str) -> bool:
    return key in (answer or "")


def run_smoke() -> dict:
    cases = []

    # ---- Case S: synthetic ----
    conS = _fixture()
    try:
        q = _q("SMOKE-S", "我的幸运数字是多少？", dt.datetime(2026, 10, 4, 15, 24, tzinfo=S.UTC))
        rec = run_condition(q, "C", "", con=conS)  # no memory doc; retrieval only
    finally:
        conS.close()
    ev = " ".join(e.get("text", "") for t in rec.get("delegations", []) for e in t.get("evidence", []))
    pay = " ".join(rec.get("tool_results", []))
    cases.append({
        "case": "S_synthetic", "question": q["text"],
        "status": [t.get("status") for t in rec.get("delegations", [])],
        "n_findings": [len(t.get("findings", [])) for t in rec.get("delegations", [])],
        "n_evidence": [len(t.get("evidence", [])) for t in rec.get("delegations", [])],
        "evidence_has_key": "7749" in ev, "payload_has_key": "7749" in pay,
        "answer": rec.get("answer", "")[:400],
        "ok": ("7749" in ev) and ("7749" in pay) and _uses(rec.get("answer", ""), "7749")
              and any(t.get("status") == "found" for t in rec.get("delegations", [])),
    })

    # ---- Case R: real data, fact absent from the memory doc + question ----
    con = store.ro()
    try:
        row = con.execute(
            "SELECT message_id, conversation_id, node_id, role, create_time, text FROM messages "
            "WHERE is_selected_branch=1 AND is_assistant_reasoning=0 AND role='user' "
            "AND message_id LIKE 'cf6f618e%' LIMIT 1").fetchone()
        key = "2106"
        when = dt.datetime.fromtimestamp(row["create_time"], S.SH) + dt.timedelta(days=3)
        q = _q("SMOKE-R", "我当时把我的 M1 Air 卖了多少钱？", when,
               context=store.conversation_thread(con, row["conversation_id"], S.epoch_of(when.isoformat()))[:6])
        rec = run_condition(q, "C", "", con=con)  # no memory doc; retrieval only
    finally:
        con.close()
    ev = " ".join(e.get("text", "") for t in rec.get("delegations", []) for e in t.get("evidence", []))
    pay = " ".join(rec.get("tool_results", []))
    cases.append({
        "case": "R_real", "question": q["text"], "expected_key": key,
        "status": [t.get("status") for t in rec.get("delegations", [])],
        "n_findings": [len(t.get("findings", [])) for t in rec.get("delegations", [])],
        "n_evidence": [len(t.get("evidence", [])) for t in rec.get("delegations", [])],
        "evidence_has_key": key in ev, "payload_has_key": key in pay,
        "answer": rec.get("answer", "")[:400],
        "ok": (key in ev) and (key in pay) and _uses(rec.get("answer", ""), key),
    })
    return {"cases": cases}
