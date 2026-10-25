"""Auxiliary grading of the custom-route answers against the frozen expectations.

Model grading is AUXILIARY only. It never overrides the raw evidence, the memory
versions, or the actual answers (all kept under run/). One request total.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parent / "run"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "letta-prototype" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_experiment as rx  # noqa: E402

SYSTEM = """你在为一次记忆系统评测做**辅助打分**。你会得到每道题的：参考答案要点（来自真实用户原话）、
标准来源判定（explicit=用户明说 / mixed=部分明说部分未定 / unknown=用户从未确定），以及被测系统的回答。

请逐题判定 verdict：
- "correct"    回答与参考答案要点一致（允许措辞不同），且没有把 unknown 当事实；
- "partial"    覆盖了一部分，或漏掉关键要点，或有轻微过度推断；
- "wrong"      与参考答案矛盾，或把 unknown 当事实，或编造了参考里没有的事实；
- "overclaim"  方向对但把不确定的、未证实的、或助手的话当成用户事实。

同时给出 flags：["overclaim"|"invented"|"missed_temporary"|"conflates_time"|"wrong_entity"|"cited_assistant_as_user"] 中适用的项。
只输出 JSON：{"grades":[{"qid":"...","verdict":"...","flags":[...],"reason":"<=40字"}]}"""


def main() -> None:
    bench = json.loads((ROOT / "shared" / "benchmark.json").read_text("utf-8"))
    exp = {}
    for cp in bench["checkpoints"]:
        for q in cp["questions"]:
            exp[(cp["checkpoint_id"], q["question"])] = q["expected"]
    raw = json.loads((RUN / "eval" / "all.json").read_text("utf-8"))

    items = []
    for rec in raw:
        qmap = {q["qid"]: q for q in rec["questions"]}
        for a in rec["answers"].get("answers", []):
            q = qmap.get(a.get("qid"))
            if not q:
                continue
            e = exp[(rec["checkpoint_id"], q["question"])]
            items.append({
                "qid": a.get("qid"), "theme": q["theme"], "type": q["type"], "question": q["question"],
                "reference_points": e["summary"], "reference_status": e["status"],
                "system_answer": a.get("answer"), "system_certainty": a.get("certainty"),
            })

    cfg = rx.make_config(RUN / "usage.json")
    Client = rx.instrument_client(cfg)
    client = Client(cfg)
    payload = client.chat([{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": json.dumps(items, ensure_ascii=False)}],
                          max_tokens=4000, temperature=0.0, response_json=True, tag="judge")
    out = client.content(payload)
    try:
        grades = json.loads(out)
    except json.JSONDecodeError:
        grades = {"_raw": out[:3000]}
    (RUN / "eval" / "judge.json").write_text(json.dumps(grades, ensure_ascii=False, indent=2), "utf-8")
    gs = grades.get("grades", [])
    print(json.dumps(grades, ensure_ascii=False, indent=2)[:4000])
    from collections import Counter
    print("verdicts:", dict(Counter(g.get("verdict") for g in gs)))


if __name__ == "__main__":
    main()
