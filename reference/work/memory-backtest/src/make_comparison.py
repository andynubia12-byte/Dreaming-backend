#!/usr/bin/env python3
"""Build the user-readable answer comparison (local only) from recorded backtest data."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
import backtest as bt  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[1] / "out" / "comparison.md"
CONDS = ["blank", "v0", "v1", "v2"]


def load_answers(suffix: str) -> dict:
    out = {}
    for t in CONDS:
        path = bt.OUT / f"answers_{t}{suffix}.jsonl"
        out[t] = {a["question_id"]: a for a in bt.read_jsonl(path)} if path.exists() else {}
    return out


def judge_map(path: Path) -> dict:
    d = json.loads(path.read_text("utf-8"))
    mp = d["mapping"]
    res = d["verdict"].get("results", {})
    out = {}
    for key, per in res.items():
        out[key] = {v: per.get(lab, {}) for lab, v in mp.get(key, {}).items()}
    return out


def section(title: str, questions: list[dict], answers: dict, judged: dict, conden: list[str]) -> list[str]:
    lines = [f"## {title}", ""]
    for q in questions:
        key = q["id"][:8]
        lines.append(f"### 问题 {key}（{q['time'][:16]}）")
        lines.append("")
        lines.append("**真实用户消息（作为问题）**：")
        lines.append("")
        lines.append("> " + q["text"].replace("\n", "\n> "))
        lines.append("")
        lines.append("| 条件 | 模型评判 overall | 幻觉标记 |")
        lines.append("|---|---|---|")
        for c in conden:
            s = judged.get(key, {}).get(c, {})
            lines.append(f"| {c} | {s.get('overall', '—')} | {s.get('hallucinated', '—')} |")
        lines.append("")
        for c in conden:
            a = answers.get(c, {}).get(q["id"], {})
            text = a.get("answer", "(缺失)").strip() or "(空回答)"
            lines.append(f"**{c} 的回答**：")
            lines.append("")
            lines.append(text)
            lines.append("")
        lines.append("---")
        lines.append("")
    return lines


def main() -> None:
    dev_q = bt.read_jsonl(bt.DATA / "dev_questions.jsonl")
    hold_q = bt.read_jsonl(bt.DATA / "holdout_questions.jsonl")
    dev_ans = load_answers("")
    hold_ans = load_answers("_holdout")
    dev_judged = judge_map(bt.OUT / "judge_batch_dev.json")
    hold_judged = judge_map(bt.OUT / "judge_holdout.json")

    lines = [
        "# 记忆回测 · 回答对照（本地阅读用）",
        "",
        "本文件含真实用户消息与模型回答（个人内容，仅本地留存，请勿外传）。",
        "条件：`blank`=无记忆；`v0`=基线提示词；`v1`/`v2`=两轮改进提示词。",
        "分数为**模型评判**（0–5），非人类客观评分；幻觉标记来自同一评判模型。",
        "",
    ]
    lines += section("开发集（2026-10-08，用于迭代）", dev_q, dev_ans, dev_judged, CONDS)
    lines += section("Holdout（2026-10-09，冻结未用于调参；只比较 blank 与 v2）", hold_q, hold_ans, hold_judged, ["blank", "v2"])
    OUTPUT.write_text("\n".join(lines), "utf-8")
    print(json.dumps({"out": str(OUTPUT), "chars": len(OUTPUT.read_text("utf-8"))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
