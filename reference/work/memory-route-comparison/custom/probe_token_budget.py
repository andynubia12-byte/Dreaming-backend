"""Probe: how many completion tokens does ONE batch consolidation need?

Calls the SAME memory_agent prompt (build_consolidate_input + CONSOLIDATE_SYSTEM)
directly with a chosen max_tokens, and reports finish_reason / completion_tokens /
whether the JSON section parses. 1 request per invocation.
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


def main() -> None:
    from lettaproto import agents, db, ideastore
    from lettaproto.deepseek import extract_json

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16000
    bid = sys.argv[2] if len(sys.argv) > 2 else "b1"
    bench = json.loads(rx.BENCH.read_text("utf-8"))
    cfg = rx.make_config(RUN / "usage.json")
    Client = rx.instrument_client(cfg)
    client = Client(cfg)
    conn = db.connect(cfg.db_path)
    b = next(x for x in bench["batches"] if x["batch_id"] == bid)
    sids = [s for mid in b["message_ids"]
            for s in next(m for m in bench["messages"] if m["id"] == mid)["source_ids"]]
    verified, rejected = agents.verify_source_ids(conn, sids, "run")
    evidence = verified
    prompt = agents.build_consolidate_input(cfg, ideastore.load(cfg, "run") if (cfg.memory_root/"run"/"idea.md").exists() else "", evidence, None, now=b["now_iso"])
    print("evidence items:", len(evidence), "prompt chars:", len(prompt))
    payload = client.chat([{"role": "system", "content": agents.CONSOLIDATE_SYSTEM},
                           {"role": "user", "content": prompt}],
                          max_tokens=n, temperature=0.2, response_json=True, tag=f"probe-{bid}")
    u = payload.get("usage") or {}
    fin = (payload.get("choices") or [{}])[0].get("finish_reason")
    text = Client.content(payload)
    print("finish:", fin, "prompt_tokens:", u.get("prompt_tokens"), "completion_tokens:", u.get("completion_tokens"))
    print("content len:", len(text))
    try:
        obj = extract_json(text)
        sec = (obj.get("section_markdown") or "")
        print("parsed OK; section chars:", len(sec), "summary:", (obj.get("summary") or "")[:80])
        print("section head:", sec[:400].replace("\n", " ⏎ "))
    except Exception as exc:  # noqa: BLE001
        print("parse failed:", type(exc).__name__, str(exc)[:200])
        print("raw head:", text[:400])
    conn.close()


if __name__ == "__main__":
    main()
