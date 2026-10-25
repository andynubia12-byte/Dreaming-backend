#!/usr/bin/env python3
"""Create the demo Letta agent and seed its MemFS from the frozen experiment.

The frozen experiment memory is treated as read-only source; we COPY it into a
brand-new agent's memory repo, so the experiment stays untouched and the demo
agent's conversation history is empty (no benchmark Q&A records carried over).
Idempotent: re-running keeps the existing demo agent.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FROZEN_MEMORY, MODEL, ROUTE, child_env, load_state, log_usage, memory_dir,
    run_oneshot, save_state,
)

SEED_FILES = ["MEMORY.md", "human.md", "persona.md"]
SEED_DIRS = []  # Personal seed directories omitted.


def create_agent() -> str:
    out = run_oneshot(["-p", "Reply with exactly: READY", "--backend", "local",
                       "--new-agent", "--model", MODEL, "--output-format", "json"])
    log_usage({"kind": "create_agent", "result": out.get("result"), "usage": out.get("usage")})
    agent = out.get("agent_id")
    if not agent:
        raise RuntimeError(f"agent creation failed: {out}")
    return agent


def seed_memory(agent_id: str) -> dict:
    dst = memory_dir(agent_id)
    if not dst.exists():
        raise RuntimeError(f"memory dir missing: {dst}")
    copied = []
    for name in SEED_FILES:
        src = FROZEN_MEMORY / name
        if src.exists():
            shutil.copy2(src, dst / name)
            copied.append(name)
    for name in SEED_DIRS:
        src = FROZEN_MEMORY / name
        target = dst / name
        if src.exists():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(src, target)
            copied.append(name + "/")
    env = child_env()
    subprocess.run(["git", "-C", str(dst), "add", "-A"], env=env, capture_output=True, text=True)
    c = subprocess.run(["git", "-C", str(dst), "commit", "-m",
                        "seed: import frozen-experiment memory into demo agent"],
                       env=env, capture_output=True, text=True)
    rev = subprocess.run(["git", "-C", str(dst), "rev-parse", "HEAD"], env=env,
                         capture_output=True, text=True)
    return {"copied": copied, "commit": (rev.stdout or "").strip()[:12],
            "commit_out": (c.stdout + c.stderr).strip()[:200]}


def main() -> None:
    state = load_state()
    if state.get("agent_id") and memory_dir(state["agent_id"]).exists():
        print(json.dumps({"ok": True, "action": "kept", "agent_id": state["agent_id"]}, ensure_ascii=False))
        return
    agent = create_agent()
    seeded = seed_memory(agent)
    state["agent_id"] = agent
    state["seeded"] = seeded
    save_state(state)
    print(json.dumps({"ok": True, "action": "created", "agent_id": agent, **seeded}, ensure_ascii=False))


if __name__ == "__main__":
    main()
