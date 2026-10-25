#!/usr/bin/env python3
"""history-bootstrap-v1 CLI: read-only ChatGPT export -> evidence / batches / retrieval import.

    python3 bootstrap.py plan  --export-dir DIR [--user-id UID] [--window month|year] [--out plan.json]
    python3 bootstrap.py build --export-dir DIR --out DIR [--user-id UID] [--window month|year]

No model/API calls. Never writes into the export dir. Creates only a NEW
retrieval_import.sqlite (matching a copy of the retrieval schema).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
from pathlib import Path

from hb_core import RETRIEVAL_SCHEMA, SH, collect


def _atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, "utf-8")
    os.replace(tmp, path)


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _as_of_epoch(value: str | None) -> float | None:
    if not value:
        return None
    parsed = dt.datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SystemExit("--as-of must be timezone-aware, e.g. 2026-10-13T20:00:00+08:00")
    return parsed.timestamp()


def _batch_message(rec: dict) -> dict:
    mid = rec.get("message_id") or f"{rec['conversation_id']}#{rec['node_id']}"
    return {"id": mid, "conversation_id": rec["conversation_id"], "role": rec["role"],
            "message_at": rec["message_at"], "text": rec["text"]}


def _write_retrieval(path: Path, convs: list[dict], kept: list[dict]) -> None:
    """A NEW sqlite matching the retrieval schema, carrying evidence-grade rows only
    (selected branch, user/assistant, non-reasoning) — never system/tool/reasoning."""
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(RETRIEVAL_SCHEMA)
        for c in convs:
            conn.execute("INSERT OR REPLACE INTO conversations VALUES(?,?,?,?,?,?,?,?)",
                         (c["conversation_id"], c["title"], c["create_time"], c["update_time"],
                          c["current_node"], c["nodes_total"], len(c["records"]), c["source_file"]))
        for r in kept:
            conn.execute("INSERT OR REPLACE INTO messages VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                         (r["conversation_id"], r["node_id"], r["message_id"], r["parent_id"], r["role"],
                          r["content_type"], r["text"], r["message_at_epoch"], 1, 0,
                          1 if r["attachments"] else 0))
        conn.commit()
    finally:
        conn.close()


def cmd_plan(args) -> int:
    plan = collect(Path(args.export_dir), user_id=args.user_id, window=args.window,
                   as_of=_as_of_epoch(args.as_of))["plan"]
    text = _dump(plan)
    if args.out:
        _atomic(Path(args.out), text + "\n")
    print(text)
    return 0 if plan["status"] == "ok" else 2


def cmd_build(args) -> int:
    export_dir = Path(args.export_dir).resolve()
    out = Path(args.out).resolve()
    if out == export_dir or export_dir in out.parents:
        raise SystemExit("refuse: --out must not be inside --export-dir (input is read-only)")
    as_of = _as_of_epoch(args.as_of)
    if as_of is None:
        raise SystemExit("build requires --as-of (tz-aware) so no future-dated batch is produced")
    data = collect(export_dir, user_id=args.user_id, window=args.window, as_of=as_of)
    plan = data["plan"]
    manifest = {
        "tool": plan["tool"], "user_id": plan["user_id"], "window": plan["window"],
        "as_of": plan["as_of"],
        "text_policy": plan["text_policy"],
        "source_files": plan["source_files"], "counts": plan["counts"],
        "branch_issues": plan["branch_issues"], "out_of_order": plan["out_of_order"],
        "duplicate_conflicts": plan["duplicate_conflicts"], "notes": plan["notes"],
        "oversized_batches": plan["oversized_batches"], "status": plan["status"], "batches": [],
    }
    out.mkdir(parents=True, exist_ok=True)
    if plan["status"] != "ok":
        _atomic(out / "manifest.json", _dump(manifest) + "\n")
        print(_dump({"status": "failed", "duplicate_conflicts": plan["duplicate_conflicts"]}))
        return 2

    lines = [json.dumps(r, ensure_ascii=False) for r in data["kept"]]
    _atomic(out / "evidence.jsonl", "\n".join(lines) + ("\n" if lines else ""))

    by_window = {w["window"]: w for w in data["windows"].values()}
    for b in plan["batches"]:
        if b["now"] > plan["as_of"]:
            raise SystemExit(f"internal error: batch {b['batch_id']} now is after --as-of")
        recs = sorted(by_window[b["window"]]["records"],
                      key=lambda r: (r["message_at_epoch"], r["branch_index"]))
        msgs = [_batch_message(r) for r in recs]
        for m in msgs:  # boundary guard: a slice must never carry future evidence
            if not (m["message_at"] < b["now"]):
                raise SystemExit(f"internal error: batch {b['batch_id']} has future evidence")
        batch = {"user_id": plan["user_id"], "batch_id": b["batch_id"], "now": b["now"],
                 "messages": msgs,
                 "_meta": {"window": b["window"], "window_end": b["now"],
                           "as_of": plan["as_of"], "clamped_by_as_of": b["clamped_by_as_of"],
                           "n_messages": len(msgs), "n_chars": b["new_chars"],
                           "note": "fixed time boundary; no message at/after now"}}
        _atomic(out / "batches" / b["file"], _dump(batch) + "\n")
        manifest["batches"].append({"batch_id": b["batch_id"], "file": f"batches/{b['file']}",
                                    "window": b["window"], "now": b["now"], "messages": len(msgs)})

    _write_retrieval(out / "retrieval_import.sqlite", data["convs"], data["kept"])
    manifest["evidence_path"] = "evidence.jsonl"
    manifest["retrieval_db"] = "retrieval_import.sqlite"
    _atomic(out / "manifest.json", _dump(manifest) + "\n")
    print(_dump({"status": manifest["status"], "user_id": manifest["user_id"],
                 "counts": manifest["counts"], "batch_count": len(manifest["batches"])}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bootstrap.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "build"):
        p = sub.add_parser(name)
        p.add_argument("--export-dir", required=True)
        p.add_argument("--user-id", default="chatgpt-export")
        p.add_argument("--window", default="month", choices=("day", "month", "year"))
        p.add_argument("--as-of", default=None,
                       help="tz-aware ISO cutoff; only messages strictly before it are kept")
        if name == "plan":
            p.add_argument("--out", default=None)
        else:
            p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "build" and not args.as_of:
        ap.error("build requires --as-of (tz-aware), e.g. --as-of 2026-10-13T20:00:00+08:00")
    return cmd_plan(args) if args.cmd == "plan" else cmd_build(args)


if __name__ == "__main__":
    raise SystemExit(main())
