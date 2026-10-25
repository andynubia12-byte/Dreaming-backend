"""Generate the user-facing answer comparison document from the recorded runs.

Reads only files produced by this rebuild (out/answers_*, out/oracle_*, out/judge_*).
"""
from __future__ import annotations

import json
from pathlib import Path

from . import settings as S

ROOT_OUT = S.HERE.parent.parent / "outputs"
LABELS = {"A": "A 仅会话", "B": "B +记忆", "C": "C +记忆+检索(新)", "O": "O +记忆+oracle(诊断)"}


def rl(p) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text("utf-8").splitlines() if l.strip()]


def _trunc(t: str, n: int) -> str:
    t = (t or "").strip().replace("\r", "")
    return t if len(t) <= n else t[:n] + f"\n…（截断，原文 {len(t)} 字）"


def _scores(judge_path: Path) -> dict:
    if not judge_path.exists():
        return {}
    d = json.loads(judge_path.read_text("utf-8"))
    mapping = d["mapping"]
    out = {c: {"ov": [], "ref": [], "noh": [], "h": 0, "u": 0} for c in ("A", "B", "C", "O")}
    for qk, labs in (d["verdict"].get("results") or {}).items():
        for lab, sc in labs.items():
            c = mapping.get(qk, {}).get(lab)
            if not c:
                continue
            out[c]["ov"].append(sc.get("overall", 0))
            out[c]["ref"].append(sc.get("reference", 0))
            out[c]["noh"].append(sc.get("no_hallucination", 0))
            out[c]["h"] += 1 if sc.get("hallucinated") else 0
            out[c]["u"] += 1 if sc.get("used_evidence") else 0
    return out


def _score_block(judge_path: Path, n: int) -> str:
    s = _scores(judge_path)
    if not s:
        return "（无判分）"
    rows = ["| 条件 | overall | reference | no_hallucination | hallucinated | 用了检索证据 |",
            "|---|---|---|---|---|---|"]
    for c in ("A", "B", "C", "O"):
        x = s[c]
        k = len(x["ov"]) or 1
        rows.append(f"| {LABELS[c]} | **{sum(x['ov'])/k:.2f}** | {sum(x['ref'])/k:.2f} | "
                    f"{sum(x['noh'])/k:.2f} | {x['h']}/{k} | {x['u']}/{k} |")
    return "\n".join(rows)


def _oracle_block(orc: dict, qid: str) -> str:
    refs = orc.get(qid) or []
    if not refs:
        return "**核验结论：该题在 `now` 之前没有可用的历史原文（真值不存在于历史，本题无检索缺口）。**"
    lines = ["**经来源核对的相关原文（真值）**", ""]
    for r in refs:
        if r.get("missing"):
            lines.append(f"- ⚠️ 未解析引用 `{r.get('source_id')}`")
            continue
        lines.append(f"- `{r.get('message_at')}` [{r.get('role')}] （{r.get('why','')}）")
        lines.append(f"  > {_trunc(r.get('text',''), 260).replace(chr(10),' ')}")
    return "\n".join(lines)


def build_comparison() -> str:
    doc = ["# 独立记忆检索 Agent · 回答对照（用户可看）", "",
           "日期：2026-10-11（Asia/Shanghai）｜模型：deepseek-flash（api.deepseek.com，单 provider）",
           "条件：A=仅当前会话；B=+长期记忆；C=+长期记忆+**新独立检索 Agent**；O=+长期记忆+**经来源核对的 oracle 原文**（诊断组）。",
           "历史原文 = 目标时刻之前、选中分支、非推理消息；`message_at` 为消息发送时刻（+08:00）。",
           "评分只是小样本模型评判，**关键事实以上方“经来源核对的相关原文”为准**。", ""]

    blocks = [
        ("一、真实检索缺口题（答案只在过去历史里，不在问题/记忆摘要）",
         "data/gap_questions.jsonl", "out/answers_gap_A-B-C-O_v2.jsonl", "out/oracle_gap.json",
         "out/judge_gap_v2b.json", 3),
        ("二、开发集 DEV 8 题（旧失败题；条件固定，仅检索路径不同）",
         "data/dev_questions.jsonl", "out/answers_dev_A-B-C-O_v2.jsonl", "out/oracle_dev.json",
         "out/judge_dev_v2c.json", 8),
        ("三、新冻结 holdout 4 题（调参期只看 id 哈希，定版一次性打开）",
         "data/holdout_questions_new.jsonl", "out/answers_holdout_new_A-B-C-O_v2.jsonl",
         "out/oracle_holdout_new.json", "out/judge_holdout_new_v2c.json", 4),
    ]
    for title, qpath, apath, opath, jpath, n in blocks:
        qs = rl(S.HERE / qpath)
        ans = rl(S.HERE / apath)
        orc = json.loads((S.HERE / opath).read_text("utf-8"))
        by = {}
        for r in ans:
            by.setdefault(r["question_id"][:8], {})[r["cond"]] = r
        doc += [f"## {title}", "", "### 评分（辅助）", "",
                _score_block(S.HERE / jpath, n), ""]
        for q in qs:
            k = q["id"][:8]
            doc += [f"### {k} ｜原文时刻 {q.get('now')}", "", "**用户消息**", "",
                    "> " + q["text"].strip().replace("\n", "\n> "), "",
                    _oracle_block(orc, k), ""]
            for c in ("A", "B", "C", "O"):
                r = by.get(k, {}).get(c, {})
                extra = ""
                if c == "C":
                    st = [t.get("status") for t in r.get("delegations", [])]
                    acc = [e.get("source_id") for e in (r.get("tool_payloads") or [{}])[0].get("evidence", [])][:4] \
                        if r.get("tool_payloads") else []
                    extra = f"\n\n*检索状态：{st}；命中来源：{acc}*"
                doc += [f"**{LABELS[c]}**", "", _trunc(r.get("answer", "（无）"), 900) + extra, ""]
            doc.append("---")
            doc.append("")
    doc += ["## 说明与边界", "",
            "- oracle 组只作诊断（判断“检索失败”还是“回答失败”），不参与自动 Agent；C 从不接收 oracle。",
            "- 旧 4 题 holdout（`data/holdout_questions.jsonl`）在上一轮**已被人看过**，本文件不把它当独立验证。",
            "- A/B/C/O 同模型、同 temperature、同会话、同记忆、同 now，只有检索路径不同。",
            "- 评分是单次模型评判、样本小；结论以“经来源核对的相关原文”与逐题回答为准，不作为分数竞赛。", ""]
    return "\n".join(doc)


def main():
    ROOT_OUT.mkdir(parents=True, exist_ok=True)
    out = ROOT_OUT / "独立记忆检索Agent-回答对照.md"
    out.write_text(build_comparison(), "utf-8")
    print(json.dumps({"file": str(out), "chars": out.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
