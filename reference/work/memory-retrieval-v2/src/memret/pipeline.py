"""Run conditions over a question set, materialise oracle evidence, and judge."""
from __future__ import annotations

import json
import random
from pathlib import Path

from . import lm
from . import prompts as P
from . import settings as S
from . import store
from .mainmodel import run_condition
from .lm import extract_json


def rl(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text("utf-8").splitlines() if l.strip()]


def wl(path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def materialise_oracle(q: dict, refs: list[dict], con) -> list[dict]:
    """Turn verified {message_id|source_id, why} references into exact original rows."""
    bound = S.epoch_of(q["time"])
    out = []
    for ref in refs or []:
        key = ref.get("source_id") or ref.get("message_id")
        rows = store.find_row(con, key, bound, ref.get("conversation_id"))
        if not rows:
            out.append({"source_id": key, "missing": True, "why": ref.get("why", ""),
                        "error": "not found before question time"})
            continue
        r = rows[0]
        out.append({"source_id": store.sid(r), "message_id": r["id"], "role": r["role"],
                    "message_at": S.now_iso(r["t"]), "time": S.utc_iso(r["t"]),
                    "text": r["text"], "why": ref.get("why", "")})
    return out


def build_oracle(qs: list[dict], refs_map: dict, con) -> dict:
    return {q["id"][:8]: materialise_oracle(q, refs_map.get(q["id"][:8], []), con)
            for q in qs if q["id"][:8] in refs_map}


def run_questions(qs: list[dict], conds: list[str], memory: str, con,
                  oracle_ev: dict | None = None) -> list[dict]:
    out = []
    for q in qs:
        qq = dict(q)
        orc = (oracle_ev or {}).get(q["id"][:8], [])
        qq["_oracle"] = orc
        for cond in conds:
            try:
                rec = run_condition(qq, cond, memory, con=con)
            except SystemExit as exc:
                out.append({"question_id": q["id"], "cond": cond, "answer": "", "error": str(exc)})
                return out
            rec["oracle_refs"] = orc
            out.append(rec)
    return out


JUDGE_SYS = (
    "You evaluate a memory-system backtest. For each question you get the real user message, a "
    "bounded excerpt of the LONG-TERM MEMORY, a bounded sample of past messages, verified ORACLE "
    "evidence (real DB rows), and ANONYMIZED candidate answers. Some candidates also include the "
    "exact evidence that candidate's retrieval returned (real DB rows, role+message_at).\n"
    "Score each answer 0-5 on: reference (correct people/events/facts vs the question), "
    "no_hallucination, overall. Add hallucinated (bool) and used_evidence (bool).\n"
    "IMPORTANT: the past sample is only a bounded slice. Do NOT mark a claim as hallucinated merely "
    "because it is absent from the sample. Mark hallucinated ONLY if the claim CONTRADICTS the "
    "provided evidence/question, or invents a person/event/date that no provided evidence supports. "
    "An honest 'I have no record of this' is NOT a hallucination. Judge substance, not length. "
    'JSON only: {"results":{"<qkey>":{"<label>":{"reference":int,"no_hallucination":int,'
    '"overall":int,"hallucinated":bool,"used_evidence":bool}}},'
    '"notes":{"<qkey>":"<one line>"}}')


def _bounded_memory(memory: str, n: int = 11000) -> str:
    memory = memory or "(none)"
    return memory if len(memory) <= n else memory[:n] + "\n…[memory doc truncated for judging]…"


def judge(qs: list[dict], answers: list[dict], memory: str, past_sample: list[dict],
          oracle_ev: dict | None, conds: list[str], seed: str = "v2") -> dict:
    by_q: dict = {}
    for r in answers:
        by_q.setdefault(r["question_id"][:8], {})[r["cond"]] = r
    packed, mapping = [], {}
    for q in qs:
        k = q["id"][:8]
        order = list(conds)
        random.Random(seed + k).shuffle(order)          # per-question anonymised label order
        lab_of = {c: f"#{i + 1}" for i, c in enumerate(order)}
        mapping[k] = {lab: c for c, lab in lab_of.items()}
        cand = {}
        for c in order:
            rec = by_q.get(k, {}).get(c, {})
            entry = {"answer": rec.get("answer", "(no answer)")}
            if c in ("C", "O"):
                ev = []
                for p in (rec.get("tool_payloads") or [])[:2]:
                    for e in (p.get("evidence") or [])[:6]:
                        ev.append({"source_id": e.get("source_id"), "role": e.get("role"),
                                   "message_at": e.get("message_at"),
                                   "text": (e.get("text") or "")[:900]})
                if ev:
                    entry["retrieved_evidence"] = ev
            cand[lab_of[c]] = entry
        item = {"qkey": k, "question": q["text"], "now": q.get("now"), "candidates": cand}
        if oracle_ev and oracle_ev.get(k):
            item["oracle_evidence"] = [
                {"source_id": r.get("source_id"), "role": r.get("role"),
                 "message_at": r.get("message_at"), "text": (r.get("text") or "")[:1200],
                 "why": r.get("why", "")} for r in oracle_ev[k] if not r.get("missing")]
        packed.append(item)
    user = ("LONG-TERM MEMORY excerpt:\n" + _bounded_memory(memory) +
            "\n\nPAST MESSAGE SAMPLE (bounded):\n" + json.dumps(past_sample, ensure_ascii=False) +
            "\n\nQUESTIONS AND ANONYMIZED ANSWERS:\n" + json.dumps(packed, ensure_ascii=False))
    p = lm.call("judge", [{"role": "system", "content": JUDGE_SYS},
                          {"role": "user", "content": user}],
                max_tokens=32000, response_json=True)
    v = extract_json(lm.content(p) or "{}")
    return {"mapping": mapping, "verdict": v}
