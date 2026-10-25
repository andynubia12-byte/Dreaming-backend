"""Bridge to the already-validated local Letta runtime (read-only dependency).

Isolation: every Letta child runs with a project-local HOME + LETTA_HOME +
LETTA_LOCAL_BACKEND_DIR, so the ambient ~/.letta is never written. The DeepSeek
key is inherited from the trusted wrapper environment only.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
from pathlib import Path
from typing import Any

from .config import Config


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def child_env(config: Config) -> dict[str, str]:
    env = dict(os.environ)
    # NOTE: HOME is intentionally NOT reassigned. The vendored, patched bundle
    # makes hardcoded home paths follow LETTA_HOME instead (see vendor.py).
    env["LETTA_HOME"] = str(config.letta_home)
    env["LETTA_LOCAL_BACKEND_DIR"] = str(config.letta_home / "lc-local-backend")
    env["LETTA_DISABLE_CRON_SCHEDULER"] = "1"
    env.setdefault("GIT_AUTHOR_NAME", "Letta Prototype")
    env.setdefault("GIT_AUTHOR_EMAIL", "noreply@letta.local")
    env.setdefault("GIT_COMMITTER_NAME", "Letta Prototype")
    env.setdefault("GIT_COMMITTER_EMAIL", "noreply@letta.local")
    return env


def run_letta(config: Config, args: list[str], *, timeout: int = 300) -> subprocess.CompletedProcess:
    config.letta_home.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        [str(config.letta_entry), *args],
        env=child_env(config),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def list_local_agents(config: Config) -> list[str]:
    agents_dir = config.letta_home / "lc-local-backend" / "agents"
    if not agents_dir.exists():
        return []
    return sorted(p.name for p in agents_dir.glob("*.json"))


def create_agent(config: Config, prompt: str = "Reply with exactly: READY") -> dict[str, Any]:
    proc = run_letta(
        config,
        ["-p", prompt, "--backend", "local", "--new-agent", "--model", config.letta_model,
         "--no-skills", "--output-format", "json"],
    )
    return _parse_result(proc)


def chat_cli(config: Config, agent_id: str, prompt: str, *, new_conversation: bool = True) -> dict[str, Any]:
    args = ["-p", prompt, "--backend", "local", "--agent", agent_id, "--no-skills", "--output-format", "json"]
    if new_conversation:
        args.insert(args.index("--no-skills"), "--new")
    proc = run_letta(config, args)
    return _parse_result(proc)


def _parse_result(proc: subprocess.CompletedProcess) -> dict[str, Any]:
    text = (proc.stdout or "").strip()
    match = re.search(r"\{.*\}\s*$", text, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            data["_returncode"] = proc.returncode
            if proc.returncode != 0:
                data["_stderr"] = (proc.stderr or "")[:400]
            return data
        except json.JSONDecodeError:
            pass
    return {"_returncode": proc.returncode, "raw_stdout": text[-800:], "raw_stderr": (proc.stderr or "")[-800:]}


def memory_dir(config: Config, agent_id: str) -> Path:
    return config.letta_home / "lc-local-backend" / "memfs" / agent_id / "memory"


def write_agent_memory(config: Config, agent_id: str, markdown: str, *, commit_message: str = "dream: update memory") -> dict[str, Any]:
    """Write the memory file inside the agent's MemFS projection and commit it."""
    mdir = memory_dir(config, agent_id)
    if not mdir.exists():
        return {"ok": False, "error": f"memory dir not found: {mdir}"}
    target = mdir / "human.md"
    enriched = (
        "---\n"
        "name: Human\n"
        'description: "Detailed long-term memory of the user, with provenance recorded in the canonical store."\n'
        "---\n\n" + markdown
    )
    # atomic-ish write: temp then replace; on failure the original file stays intact
    tmp = mdir / ".human.md.tmp"
    tmp.write_text(enriched, "utf-8")
    tmp.replace(target)
    commit = subprocess.run(
        ["git", "-C", str(mdir), "add", "-A"],
        env=child_env(config), capture_output=True, text=True,
    )
    commit2 = subprocess.run(
        ["git", "-C", str(mdir), "commit", "-m", commit_message, "--allow-empty"],
        env=child_env(config), capture_output=True, text=True,
    )
    rev = subprocess.run(["git", "-C", str(mdir), "rev-parse", "HEAD"], env=child_env(config), capture_output=True, text=True)
    return {
        "ok": commit2.returncode == 0,
        "chars": len(enriched),
        "path": str(target),
        "commit": (rev.stdout or "").strip()[:12],
        "commit_out": (commit2.stdout + commit2.stderr).strip()[:200],
    }


def compiled_system_prompt(config: Config, agent_id: str, conversation_id: str | None = None) -> dict[str, Any] | None:
    conv_root = config.letta_home / "lc-local-backend" / "conversations"
    if not conv_root.exists():
        return None
    best: Path | None = None
    best_mtime = -1.0
    for path in conv_root.glob("*/system-prompt.json"):
        try:
            data = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        conv = path.parent / "conversation.json"
        agent_ok = agent_id.replace("-", "") in str(path.parent.name)
        if agent_id and not agent_ok:
            # fall back to scanning conversation.json for the agent id
            if conv.exists():
                try:
                    if agent_id not in json.dumps(json.loads(conv.read_text("utf-8"))):
                        continue
                except (OSError, json.JSONDecodeError):
                    continue
        mtime = path.stat().st_mtime
        if mtime > best_mtime:
            best, best_mtime = path, mtime
    if not best:
        return None
    data = json.loads(best.read_text("utf-8"))
    return {"path": str(best), "content_chars": len(data.get("content") or ""), "data": data}


def start_server(config: Config, port: int | None = None) -> dict[str, Any]:
    port = port or free_port()
    proc = subprocess.Popen(
        [str(config.letta_entry), "server", "--backend", "local",
         "--listen", f"ws://127.0.0.1:{port}", "--openai-api"],
        env=child_env(config),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return {"port": port, "url": f"http://127.0.0.1:{port}/v1", "pid": proc.pid, "process": proc}
