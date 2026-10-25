#!/usr/bin/env python3
"""Letta-native demo server on 127.0.0.1:8789.

Chat is served by a real Letta agent (deepseek/deepseek-flash) through a
long-lived bidirectional headless session, so the native post-turn reflection
("Dreaming") stays alive and keeps updating THIS demo's memory copy. Nothing
here re-implements chat or retrieval: memory reads/searches are plain reads of
the agent's real MemFS files, and memory history is the agent's real git log.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    LETTA_ENTRY, MODEL, ROUTE, WRAPPER, child_env, load_state, log_usage, memory_dir,
)

PORT = 8789
TZ = timezone(timedelta(hours=8))
UI = ROUTE / "ui" / "trial.html"


def now_iso() -> str:
    return datetime.now(TZ).isoformat(timespec="seconds")


class LettaSession:
    """One long-lived real Letta session (bidi headless) for the demo agent."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.lock = threading.Lock()
        self.cond = threading.Condition()
        self.proc: subprocess.Popen | None = None
        self.results: list[dict] = []
        self.conversation_id: str | None = None
        self.turns = 0
        self._start(new=True)

    def _start(self, new: bool) -> None:
        args = ["--backend", "local", "--agent", self.agent_id, "--model", MODEL,
                "--reflection-trigger", "step-count", "--reflection-step-count", "1",
                "--input-format", "stream-json", "--output-format", "stream-json"]
        if new:
            args.insert(2, "--new")
        cmd = ["node", str(WRAPPER), "run", "default", "--", "node", str(LETTA_ENTRY), *args]
        self.proc = subprocess.Popen(cmd, env=child_env(), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, bufsize=1)
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._drain_err, daemon=True).start()
        self.conversation_id = None
        self.turns = 0

    def _read(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            with self.cond:
                if ev.get("type") == "system" and ev.get("subtype") == "init":
                    self.conversation_id = ev.get("conversation_id") or self.conversation_id
                if ev.get("type") == "result":
                    self.results.append(ev)
                    self.turns += 1
                    self.cond.notify_all()

    def _drain_err(self) -> None:
        assert self.proc and self.proc.stderr
        for _ in self.proc.stderr:
            pass

    def ask(self, message: str, timeout: float = 210.0) -> dict | None:
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self._start(new=False)
            with self.cond:
                n = len(self.results)
            try:
                assert self.proc.stdin
                self.proc.stdin.write(json.dumps(
                    {"type": "user", "message": {"content": message}}, ensure_ascii=False) + "\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, ValueError):
                self._start(new=False)
                assert self.proc.stdin
                self.proc.stdin.write(json.dumps(
                    {"type": "user", "message": {"content": message}}, ensure_ascii=False) + "\n")
                self.proc.stdin.flush()
            deadline = time.time() + timeout
            with self.cond:
                while len(self.results) <= n:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        return None
                    self.cond.wait(remaining)
                return self.results[n]

    def reset(self) -> None:
        with self.lock:
            try:
                if self.proc and self.proc.stdin:
                    self.proc.stdin.close()
            except Exception:
                pass
            try:
                if self.proc:
                    self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.results = []
            self._start(new=True)


def list_memory_files(agent_id: str) -> list[Path]:
    m = memory_dir(agent_id)
    if not m.exists():
        return []
    return sorted(p for p in m.rglob("*") if p.is_file() and ".git" not in p.parts)


def memory_text(agent_id: str) -> str:
    m = memory_dir(agent_id)
    out = [f"# 记忆目录（真实 MemFS，Agent {agent_id}）", ""]
    for p in list_memory_files(agent_id):
        rel = p.relative_to(m)
        out.append(f"\n===== {rel} =====\n")
        out.append(p.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(out)


def memory_git_log(agent_id: str, limit: int = 30) -> list[dict]:
    m = memory_dir(agent_id)
    r = subprocess.run(["git", "-C", str(m), "log", f"-{limit}",
                        "--pretty=%H|%ad|%s", "--date=iso"],
                       env=child_env(), capture_output=True, text=True)
    commits = []
    for line in r.stdout.splitlines():
        parts = line.split("|", 2)
        if len(parts) == 3:
            commits.append({"hash": parts[0][:12], "date": parts[1], "subject": parts[2]})
    return commits


def query_memory(agent_id: str, question: str, limit: int = 12) -> list[dict]:
    """Honest keyword search over the real memory files (no model call)."""
    tokens = [t for t in question.replace("，", " ").replace(",", " ").split() if t]
    if not tokens:
        return []
    m = memory_dir(agent_id)
    hits = []
    for p in list_memory_files(agent_id):
        rel = str(p.relative_to(m))
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, line in enumerate(lines, 1):
            if any(t in line for t in tokens):
                hits.append({"file": rel, "line": i, "excerpt": line.strip()[:280]})
                if len(hits) >= limit:
                    return hits
    return hits


class Handler(BaseHTTPRequestHandler):
    server_version = "LettaNativeDemo/1.0"
    session: LettaSession
    agent_id: str

    def log_message(self, *args):  # silence default logging (no secrets)
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n == 0:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return {}

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(200, UI.read_bytes(), "text/html; charset=utf-8")
        elif path == "/health":
            m = memory_dir(self.agent_id)
            chars = sum(p.stat().st_size for p in list_memory_files(self.agent_id)) if m.exists() else 0
            commits = memory_git_log(self.agent_id, 1000)
            self._json(200, {
                "ok": True, "now": now_iso(), "agent_id": self.agent_id,
                "memory": {"chars": chars, "files": len(list_memory_files(self.agent_id))},
                "dreaming": {"enabled": True, "commits": len(commits)},
                "session": {"conversation_id": self.session.conversation_id, "turns": self.session.turns},
            })
        elif path == "/idea":
            self._send(200, memory_text(self.agent_id).encode("utf-8"), "text/plain; charset=utf-8")
        elif path == "/tasks":
            commits = memory_git_log(self.agent_id, 25)
            self._json(200, {"count": len(commits), "commits": commits})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        body = self._body()
        if path == "/chat":
            message = (body.get("message") or "").strip()
            if not message:
                self._json(400, {"error": "empty message"})
                return
            started = time.time()
            res = self.session.ask(message)
            if not res:
                self._json(504, {"error": "Letta 会话超时，请重试或点重置会话。"})
                return
            answer = res.get("result") or ""
            usage = res.get("usage")
            log_usage({"kind": "chat", "now": now_iso(), "conversation_id": res.get("conversation_id"),
                       "elapsed_s": round(time.time() - started, 2), "usage": usage,
                       "answer_chars": len(answer)})
            commits = memory_git_log(self.agent_id, 1000)
            self._json(200, {"answer": answer, "conversation_id": res.get("conversation_id"),
                             "turn": self.session.turns, "usage": usage,
                             "dreaming_commits": len(commits), "now": now_iso()})
        elif path == "/query":
            q = (body.get("question") or body.get("message") or "").strip()
            self._json(200, {"question": q, "hits": query_memory(self.agent_id, q)})
        elif path == "/update":
            commits = memory_git_log(self.agent_id, 3)
            self._json(200, {"mode": "automatic",
                             "detail": "原生后台 Dreaming 已开启：每轮对话后 Letta 会自动运行反思子 Agent 整理这份试用记忆副本。",
                             "recent_commits": commits})
        elif path == "/reset":
            try:
                self.session.reset()
            except Exception as e:
                self._json(500, {"error": str(e)})
                return
            self._json(200, {"ok": True, "conversation_id": self.session.conversation_id})
        else:
            self._json(404, {"error": "not found"})


def main() -> None:
    state = load_state()
    agent_id = state.get("agent_id")
    if not agent_id:
        raise SystemExit("no demo agent; run scripts/setup_agent.py first")
    Handler.agent_id = agent_id
    Handler.session = LettaSession(agent_id)
    httpd = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    log_usage({"kind": "server_start", "now": now_iso(), "agent_id": agent_id, "port": PORT})

    def _shutdown(signum, _frame):
        try:
            if Handler.session.proc:
                Handler.session.proc.terminate()
        except Exception:
            pass
        raise SystemExit(0)

    import signal
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
