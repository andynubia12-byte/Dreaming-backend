"""Command-line entry point.

    implementation/run_update.py --state-dir <dir> --batch <JSON> --output <JSON path>

`--batch` accepts a path to a batch JSON file or inline JSON. The result JSON is
written to `--output` and echoed (abridged) to stdout. Real model calls need the
trusted wrapper; without DEEPSEEK_API_KEY the engine fails closed with no writes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .engine import Engine
from .model import DeepSeekModel
from .prompts import load_prompts, prompt_hashes
from .store import Store
from .timeutil import TimeError, parse_iso


def _load_batch(spec: str) -> dict:
    text = spec.strip()
    if text.startswith("{"):
        return json.loads(text)
    return json.loads(Path(spec).read_text("utf-8"))


def _write(path: str, obj: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), "utf-8")


def _positive_int(value: str) -> int:
    """argparse type for a strictly positive integer.

    A NON-positive or non-numeric value raises here, so argparse aborts with a
    usage error BEFORE any store is opened or any model request is issued.
    """
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {parsed}")
    return parsed


def _dry_run(batch: dict, prompts: dict) -> dict:
    uid, bid, now = batch.get("user_id"), batch.get("batch_id"), batch.get("now")
    out: dict = {"status": "ok", "user_id": uid, "batch_id": bid, "dry_run": True,
                 "prompt_hashes": prompt_hashes(prompts)}
    try:
        parsed_now = parse_iso(now)
    except TimeError as exc:
        out.update(status="rejected_time", error=str(exc))
        return out
    for m in batch.get("messages") or []:
        try:
            at = parse_iso(m.get("message_at"))
        except TimeError as exc:
            out.update(status="rejected_time", error=str(exc), rejected_message=m.get("id"))
            return out
        if at > parsed_now:
            out.update(status="rejected_future", rejected_message=m.get("id"),
                       error=f"message {m.get('id')} is after now")
            return out
    out["messages"] = len(batch.get("messages") or [])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="run_update.py",
                                 description="Isolated long-term memory maintenance v1")
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--batch", required=True, help="path to batch JSON, or inline JSON")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--budget", type=int, default=100_000)
    ap.add_argument("--output-tokens", type=_positive_int, default=16_000,
                    help="max output tokens per model call (positive integer; "
                         "passed through unchanged, no silent clamp)")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate the batch only; no model call, no writes")
    args = ap.parse_args(argv)

    prompts = load_prompts()
    try:
        batch = _load_batch(args.batch)
    except Exception as exc:  # noqa: BLE001
        result = {"status": "error", "error": f"bad batch: {type(exc).__name__}: {exc}"}
        _write(args.output, result)
        print(json.dumps(result, ensure_ascii=False))
        return 3

    if args.dry_run:
        result = _dry_run(batch, prompts)
        _write(args.output, result)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["status"] == "ok" else 2

    store = Store(Path(args.state_dir))
    model = DeepSeekModel(ledger_dir=Path(args.state_dir) / "ledger",
                          model=args.model, max_requests=args.budget,
                          max_output_tokens=args.output_tokens)
    engine = Engine(store, model, prompts)
    result = engine.run_batch(batch)
    _write(args.output, result)
    print(json.dumps({k: result.get(k) for k in (
        "status", "user_id", "version", "summary_path", "memory_path", "evidence_path",
        "claims_applied", "expired", "model_calls")}, ensure_ascii=False))
    return 0 if result.get("status") in ("ok", "noop", "duplicate") else 2


if __name__ == "__main__":
    raise SystemExit(main())
