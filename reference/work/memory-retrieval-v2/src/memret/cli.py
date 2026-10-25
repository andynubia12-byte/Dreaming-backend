"""Command line for the v2 retrieval rebuild.

Subcommands:
  selftest                 local regression tests (no API)
  smoke                    small closed-loop acceptance (synthetic + real)
  oracle   --questions     materialise verified oracle refs into exact rows (no API)
  run      --questions     run conditions A/B/C/[O] over a question set (API)
  probe    --questions     retrieval-only diagnostic trace per question (API)
  judge    --answers       judge an answers file (API)
  usage                    print the independent ledger + local-tool counts
"""
from __future__ import annotations

import argparse
import json
import unittest
from pathlib import Path

from . import lm
from . import settings as S
from . import store
from .agent import run_memory_agent
from .pipeline import build_oracle, judge, materialise_oracle, rl, run_questions, wl


def _memory(tag: str) -> str:
    return (S.OUT / f"memory_{tag}.md").read_text("utf-8") if tag else ""


def cmd_selftest(_a):
    here = Path(__file__).resolve().parents[2]
    loader = unittest.TestLoader()
    suite = loader.discover(str(here / "tests"))
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if res.wasSuccessful() else 1)


def cmd_smoke(a):
    from .smoke import run_smoke
    out = run_smoke()
    (S.OUT / "smoke.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    ok = all(c.get("ok") for c in out["cases"])
    print(json.dumps({"ok": ok, "cases": [{k: c.get(k) for k in ("case", "status", "n_findings",
          "n_evidence", "used")} for c in out["cases"]]}, ensure_ascii=False))
    raise SystemExit(0 if ok else 1)


def cmd_oracle(a):
    qs = rl(a.questions)
    refs = json.loads(Path(a.refs).read_text("utf-8")) if a.refs else {}
    con = store.ro()
    try:
        ev = build_oracle(qs, refs, con)
    finally:
        con.close()
    out = S.OUT / f"oracle_{a.label}.json"
    out.write_text(json.dumps(ev, ensure_ascii=False, indent=2), "utf-8")
    missing = sum(1 for v in ev.values() for r in v if r.get("missing"))
    print(json.dumps({"file": out.name, "questions": len(ev), "missing_refs": missing}, ensure_ascii=False))


def cmd_run(a):
    qs = rl(a.questions)
    memory = _memory(a.memory)
    oracle_ev = None
    if a.oracle:
        oracle_ev = json.loads(Path(a.oracle).read_text("utf-8"))
    con = store.ro()
    try:
        recs = run_questions(qs, a.conds, memory, con, oracle_ev=oracle_ev)
    finally:
        con.close()
    name = f"answers_{a.label}_{'-'.join(a.conds)}{a.suffix}.jsonl"
    wl(S.OUT / name, recs)
    print(json.dumps({"file": name, "n": len(recs)}, ensure_ascii=False))


def cmd_probe(a):
    """Retrieval-only: an automatic delegation built from the question text.

    This is a DIAGNOSTIC of the retrieval path (did we find sources?), not a
    scored answer condition. It never sees oracle data.
    """
    qs = rl(a.questions)
    memory = _memory(a.memory)
    con = store.ro()
    out = []
    try:
        for q in qs:
            delg = {"question": q["text"], "now_iso": q["now"],
                    "session_context": list(q.get("context", []))[-8:],
                    "memory_excerpt": "", "gaps": [q["text"][:200]],
                    "keywords": [], "entities": [], "time_hint": "", "known": ""}
            trace = run_memory_agent(delg, S.epoch_of(q["time"]), con, max_tool_turns=a.turns)
            out.append({"question_id": q["id"], "text": q["text"][:180],
                        "status": trace["status"], "n_findings": len(trace["findings"]),
                        "n_evidence": len(trace["evidence"]), "n_sources": trace["n_sources"],
                        "steps": [{"tool": s["tool"], "args": s["args"], "n": s["n_results"]}
                                  for s in trace["steps"]],
                        "evidence": [{"source_id": e["source_id"], "role": e["role"],
                                      "message_at": e["message_at"], "text": e["text"][:300]}
                                     for e in trace["evidence"][:6]],
                        "note": trace["note"]})
    finally:
        con.close()
    name = f"probe_{a.label}{a.suffix}.json"
    (S.OUT / name).write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({"file": name, "statuses": [o["status"] for o in out]}, ensure_ascii=False))


def cmd_judge(a):
    qs = rl(a.questions)
    ans = rl(a.answers)
    memory = _memory(a.memory)
    oracle_ev = json.loads(Path(a.oracle).read_text("utf-8")) if a.oracle else None
    # bounded, deterministic past-message sample (first/last of the memory window)
    sample = rl(S.DATA / "memory_msgs.jsonl")
    past = sample[:40] + sample[-40:]
    res = judge(qs, ans, memory, past, oracle_ev, a.conds)
    out = {"mapping": res["mapping"], "verdict": res["verdict"]}
    name = f"judge_{a.label}{a.suffix}.json"
    (S.OUT / name).write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({"file": name, "ok": bool(res["verdict"].get("results"))}, ensure_ascii=False))


def cmd_usage(_a):
    led = S.ledger()
    local = {"search_history": 0, "read_message": 0, "expand": 0}
    kinds: dict = {}
    if S.CALL_LOG.exists():
        for line in S.CALL_LOG.read_text("utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("type") == "local":
                local[e.get("tool")] = local.get(e.get("tool"), 0) + 1
            else:
                kinds[e.get("kind")] = kinds.get(e.get("kind"), 0) + 1
    print(json.dumps({"ledger": led, "api_calls_by_kind": kinds, "local_tool_calls": local},
                     ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    s.add_parser("selftest").set_defaults(f=cmd_selftest)
    s.add_parser("smoke").set_defaults(f=cmd_smoke)
    s.add_parser("usage").set_defaults(f=cmd_usage)
    o = s.add_parser("oracle"); o.add_argument("--questions", required=True)
    o.add_argument("--refs", required=True); o.add_argument("--label", required=True)
    o.set_defaults(f=cmd_oracle)
    r = s.add_parser("run"); r.add_argument("--questions", required=True)
    r.add_argument("--conds", nargs="+", required=True); r.add_argument("--memory", default="t1")
    r.add_argument("--oracle", default=None); r.add_argument("--label", required=True)
    r.add_argument("--suffix", default=""); r.set_defaults(f=cmd_run)
    pr = s.add_parser("probe"); pr.add_argument("--questions", required=True)
    pr.add_argument("--memory", default="t1"); pr.add_argument("--turns", type=int, default=4)
    pr.add_argument("--label", required=True); pr.add_argument("--suffix", default="")
    pr.set_defaults(f=cmd_probe)
    j = s.add_parser("judge"); j.add_argument("--questions", required=True)
    j.add_argument("--answers", required=True); j.add_argument("--conds", nargs="+", required=True)
    j.add_argument("--memory", default="t1"); j.add_argument("--oracle", default=None)
    j.add_argument("--label", required=True); j.add_argument("--suffix", default="")
    j.set_defaults(f=cmd_judge)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
