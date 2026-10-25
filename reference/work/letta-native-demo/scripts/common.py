#!/usr/bin/env python3
"""Shared helpers for the Letta-native demo (real Letta, real time).

Isolation: every child runs with its own LETTA_HOME / local backend dir, and the
vendored bundle makes hardcoded ~/.letta paths follow LETTA_HOME. HOME is never
reassigned. Credentials come only from the trusted wrapper. Real machine time is
used (no LETTA_TIME_OFFSET_MS), so `now` is the true Asia/Shanghai time.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

ROUTE = Path(__file__).resolve().parent.parent
LETTA_ENTRY = ROUTE / "runtime" / "vendor" / "letta-code" / "letta.js"
LETTA_HOME = ROUTE / "letta_home"
LOCAL_BACKEND_DIR = LETTA_HOME / "lc-local-backend"
WRAPPER = Path("/Users/andy/.codex/skills/codex-deepseek-subagent/scripts/credential-ui/src/profile.ts")
MODEL = "deepseek/deepseek-flash"
STATE = ROUTE / "runs" / "demo_state.json"
USAGE = ROUTE / "logs" / "usage.jsonl"

# frozen-experiment memory (read-only source; we copy, never modify)
FROZEN_MEMORY = ROUTE / "private-input" / "frozen-memory"


def child_env() -> dict[str, str]:
    env = dict(os.environ)
    env["LETTA_HOME"] = str(LETTA_HOME)
    env["LETTA_LOCAL_BACKEND_DIR"] = str(LOCAL_BACKEND_DIR)
    env["LETTA_DISABLE_CRON_SCHEDULER"] = "1"
    env["LETTA_TRANSCRIPT_ROOT"] = str(LETTA_HOME / "transcripts")
    env["LETTA_SCRATCHPAD"] = str(LETTA_HOME / "tmp" / "scratch")
    env["LETTA_ARTIFACTS_DIR"] = str(LETTA_HOME / "artifacts")
    # NOTE: deliberately NO LETTA_TIME_OFFSET_MS -> real Asia/Shanghai now.
    env.pop("LETTA_TIME_OFFSET_MS", None)
    env.setdefault("GIT_AUTHOR_NAME", "Letta Demo")
    env.setdefault("GIT_AUTHOR_EMAIL", "noreply@letta.local")
    env.setdefault("GIT_COMMITTER_NAME", "Letta Demo")
    env.setdefault("GIT_COMMITTER_EMAIL", "noreply@letta.local")
    return env


def log_usage(entry: dict) -> None:
    USAGE.parent.mkdir(parents=True, exist_ok=True)
    with USAGE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def parse_json_result(proc: subprocess.CompletedProcess) -> dict:
    text = (proc.stdout or "").strip()
    m = re.search(r"\{.*\}\s*$", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            d["_returncode"] = proc.returncode
            return d
        except json.JSONDecodeError:
            pass
    return {"_returncode": proc.returncode, "raw_stdout": text[-1200:], "raw_stderr": (proc.stderr or "")[-1200:]}


def run_oneshot(args: list[str], *, timeout: int = 300) -> dict:
    LETTA_HOME.mkdir(parents=True, exist_ok=True)
    cmd = ["node", str(WRAPPER), "run", "default", "--", "node", str(LETTA_ENTRY), *args]
    proc = subprocess.run(cmd, env=child_env(), capture_output=True, text=True, timeout=timeout)
    return parse_json_result(proc)


def memory_dir(agent_id: str) -> Path:
    return LOCAL_BACKEND_DIR / "memfs" / agent_id / "memory"


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
