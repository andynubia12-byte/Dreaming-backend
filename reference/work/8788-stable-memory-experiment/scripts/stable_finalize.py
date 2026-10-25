"""Replay the summariser's saved tool calls locally (deterministic, no model cost),
then make ONE Flash call to turn the gathered evidence into the stable-layer Markdown.

This avoids re-paying for retrieval when only the final formatting step failed.
Writes stable/stable.md, stable/sources.json (sidecar) and raw/finalize_*.json.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEMO = REPO / "work" / "generated-memory-demo"
EXP = REPO / "work" / "8788-stable-memory-experiment"
SNAPSHOT = EXP / "snapshots" / "canonical-8788-20261012.sqlite"

sys.path.insert(0, str(EXP / "scripts"))
sys.path.insert(0, str(DEMO / "src"))
from lettaproto.deepseek import DeepSeekClient  # noqa: E402
from lettaproto.config import load_config  # noqa: E402
from lettaproto.retrieval import TrialRetrieval, SHANGHAI  # noqa: E402
import stable_tools  # noqa: E402  (same scripts/ dir)

FINAL = """停止调用工具。只根据上面工具返回过的证据，输出这份"稳定记忆层"的 Markdown 正文（不要 JSON，不要输出任何解释或思考）。

结构固定为三节：
## A. 稳定背景
## B. 交流偏好
## C. 有时间依据的当前状态
再加一节：
## D. 未找到 / 不确定

硬要求：
- 每一条结论单独一行，并在行内附上来源 id（如 msg:<...>#<...> 或 chat#<id> 或"来源：..."）。
- 只有 role=user 的消息算用户事实；助手消息只能作为线索并标注。
- C 节每条必须有日期，并标"当时/现在/不确定"；临时情绪不写成长期状态。
- 找不到的写进 D 节，不要编造、不要用常识补全。
- 直接给 Markdown，不要前言，不要固定字数。"""

CIT_RE = re.compile(r"(msg:[0-9a-zA-Z_-]+#[0-9a-zA-Z_-]+|chat#\d+)")

# Targeted, generic retrieval aimed at the user's EXPLICIT corrections of how the AI
# should talk to them. No personal names are hard-coded; these are communication topics.
SUPPLEMENTARY = [
    {"query": "用户明确要求的提问方式", "keywords": ["提问", "要求"]},
    {"query": "用户明确要求的建议方式", "keywords": ["建议", "要求"]},
    {"query": "用户明确要求的界面呈现方式", "keywords": ["界面", "呈现"]},
    {"query": "用户明确纠正的表达方式", "keywords": ["表达", "纠正"]},
    {"query": "用户要求长期保留的交流偏好", "keywords": ["偏好", "记住"]},
]


def to_epoch(iso: str) -> float:
    return dt.datetime.fromisoformat(iso).timestamp()


def main() -> None:
    config = load_config()
    conn = sqlite3.connect(f"file:{SNAPSHOT}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    tr = json.loads((EXP / "stable" / "summarizer_transcript.json").read_text("utf-8"))
    now = to_epoch(tr["now"])
    r = TrialRetrieval(conn, "prod", now)
    now_iso = tr["now"]

    start = (
        f"SIMULATED NOW: {now_iso} (Asia/Shanghai, +08:00). 只有此刻之前的历史存在。\n"
        "请开始：先做少量有针对性的检索，覆盖 ①用户基本背景 ②用户对AI交流方式的明确要求/纠正 "
        "③用户的长期兴趣 ④重要关系的当前状态；然后输出规定的 JSON。"
    )
    msgs = [{"role": "system", "content": stable_tools.SYSTEM}, {"role": "user", "content": start}]
    turns: dict[int, list[dict]] = {}
    for s in tr["steps"]:
        turns.setdefault(s["turn"], []).append(s)
    for turn in sorted(turns):
        calls = []
        results = []
        for i, s in enumerate(turns[turn]):
            cid = f"call_{turn}_{i}"
            calls.append({"id": cid, "type": "function",
                          "function": {"name": s["tool"], "arguments": json.dumps(s["args"], ensure_ascii=False)}})
            results.append((cid, stable_tools.dispatch(s["tool"], s["args"], r)))
        msgs.append({"role": "assistant", "content": "", "tool_calls": calls})
        for cid, res in results:
            msgs.append({"role": "tool", "tool_call_id": cid, "content": json.dumps(res, ensure_ascii=False)})

    # supplementary targeted preference retrieval (local BM25, no model cost)
    sup_calls, sup_results = [], []
    for i, s in enumerate(SUPPLEMENTARY):
        cid = f"call_sup_{i}"
        sup_calls.append({"id": cid, "type": "function",
                          "function": {"name": "search_history",
                                       "arguments": json.dumps({**s, "limit": 6}, ensure_ascii=False)}})
        sup_results.append((cid, stable_tools.dispatch("search_history", s, r)))
    msgs.append({"role": "assistant", "content": "", "tool_calls": sup_calls})
    for cid, res in sup_results:
        msgs.append({"role": "tool", "tool_call_id": cid, "content": json.dumps(res, ensure_ascii=False)})
    msgs.append({"role": "user", "content": FINAL})

    client = DeepSeekClient(config)
    payload = client.chat(msgs, max_tokens=16000, temperature=0.0, model=config.chat_model)
    finish = (payload.get("choices") or [{}])[0].get("finish_reason")
    markdown = (DeepSeekClient.content(payload) or "").strip()
    (EXP / "raw").mkdir(parents=True, exist_ok=True)
    (EXP / "raw" / "finalize_call.json").write_text(
        json.dumps({"finish_reason": finish, "messages_chars": sum(len(m.get("content") or "") for m in msgs),
                    "markdown_chars": len(markdown), "usage": client.usage.to_dict()},
                   ensure_ascii=False, indent=2), "utf-8")

    (EXP / "stable" / "stable.md").write_text(markdown + "\n", "utf-8")
    sidecar = []
    for line in markdown.splitlines():
        sids = CIT_RE.findall(line)
        if sids and line.strip() and not line.strip().startswith("#"):
            sidecar.append({"claim": line.strip(), "source_ids": sorted(set(sids))})
    (EXP / "stable" / "sources.json").write_text(
        json.dumps({"now": now_iso, "model": config.chat_model, "finish_reason": finish,
                    "claims": sidecar, "n_claims": len(sidecar)}, ensure_ascii=False, indent=2), "utf-8")
    print("FINISH", finish, "MARKDOWN_CHARS", len(markdown), "CLAIMS", len(sidecar), flush=True)
    print("USAGE", json.dumps(client.usage.to_dict()), flush=True)


if __name__ == "__main__":
    main()
