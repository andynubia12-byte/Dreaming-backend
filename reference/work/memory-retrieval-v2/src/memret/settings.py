"""Paths, simulated-time config, and the independent usage ledger / call log.

All model calls go only to api.deepseek.com / deepseek-flash through the
existing trusted credential wrapper. The ledger here is SEPARATE from the old
backtest's runs/usage.json (never reset, never merged).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
ROOT = HERE.parent
DB = ROOT / "letta-prototype" / "db" / "canonical.sqlite"
DATA, OUT, RUNS = HERE / "data", HERE / "out", HERE / "runs"
USAGE = Path(os.environ.get("MEMRET_USAGE_FILE", str(RUNS / "usage.json")))
CALL_LOG = Path(os.environ.get("MEMRET_CALL_LOG", str(RUNS / "calls.jsonl")))

SH = dt.timezone(dt.timedelta(hours=8), "Asia/Shanghai")
UTC = dt.timezone.utc

MODEL = "deepseek-flash"
ANSWER_TEMPERATURE = 0.6

# Simulated batch/checkpoint times (no real "today" is ever used).
T_MEM_START = dt.datetime(2026, 9, 20, tzinfo=UTC)
CUTOFF = dt.datetime(2026, 10, 4, tzinfo=UTC)
T_DEV = dt.datetime(2026, 10, 7, tzinfo=UTC)
T_HOLD = dt.datetime(2026, 10, 10, tzinfo=UTC)

# Generous engineering ceilings (user lifted the old 120/2M cap). These only
# ever stop a runaway loop; they never truncate a required acceptance run.
MAX_CALLS = int(os.environ.get("MEMRET_MAX_CALLS", "1000000"))
MAX_INPUT = int(os.environ.get("MEMRET_MAX_INPUT", "1000000000"))

_LOCK = threading.Lock()


def now_iso(epoch: float) -> str:
    """Simulated 'now' for a message/turn: Asia/Shanghai with explicit +08:00."""
    return dt.datetime.fromtimestamp(epoch, SH).isoformat()


def utc_iso(epoch: float) -> str:
    return dt.datetime.fromtimestamp(epoch, UTC).isoformat()


def epoch_of(iso_str: str) -> float:
    return dt.datetime.fromisoformat(iso_str).timestamp()


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def ensure_dirs() -> None:
    for p in (DATA, OUT, RUNS):
        p.mkdir(parents=True, exist_ok=True)


def ledger() -> dict:
    if USAGE.exists():
        return json.loads(USAGE.read_text("utf-8"))
    return {"requests": 0, "input_tokens": 0, "output_tokens": 0, "errors": 0}


def bump_ledger(prompt_tokens: int, completion_tokens: int, error: bool = False) -> dict:
    with _LOCK:
        led = ledger()
        led["requests"] = led.get("requests", 0) + 1
        led["input_tokens"] = led.get("input_tokens", 0) + int(prompt_tokens or 0)
        led["output_tokens"] = led.get("output_tokens", 0) + int(completion_tokens or 0)
        if error:
            led["errors"] = led.get("errors", 0) + 1
        USAGE.parent.mkdir(parents=True, exist_ok=True)
        USAGE.write_text(json.dumps(led, ensure_ascii=False, indent=2), "utf-8")
        return led


def log_event(event: dict) -> None:
    """Append one line to the per-call log (API *and* local tools). Locked."""
    with _LOCK:
        CALL_LOG.parent.mkdir(parents=True, exist_ok=True)
        with CALL_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def guard(kind: str) -> None:
    led = ledger()
    if led.get("requests", 0) >= MAX_CALLS:
        raise SystemExit(f"budget guard ({kind}): requests {led['requests']} >= {MAX_CALLS}")
    if led.get("input_tokens", 0) >= MAX_INPUT:
        raise SystemExit(f"budget guard ({kind}): input {led['input_tokens']} >= {MAX_INPUT}")
