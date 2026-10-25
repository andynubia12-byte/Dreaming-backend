"""Minimal local HTTP API (127.0.0.1 only).

GET  /            -> trial page (HTML)
GET  /health      -> status
GET  /idea        -> full working memory text
GET  /tasks       -> background memory-task queue state
POST /query       -> local keyword search over idea + history (no model)
POST /chat        -> agentic chat (full idea + recent context; tool loop)
POST /search      -> delegate a search to the Memory Agent
POST /update      -> enqueue a background memory task (fast; processed by the worker)

There is deliberately NO long-lived lock wrapped around whole POSTs: chat/search are
never blocked by consolidation, and consolidation is serialized inside the worker.
"""

from __future__ import annotations

import json
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

from . import db, ideastore
from .agents import AGENT_VERSION, memory_agent_search
from .chat import EmptyModelResponse, chat_agentic
from .config import Config
from .context import assemble_memory
from .deepseek import BudgetExceeded, DeepSeekClient, MissingCredential
from .query import query as run_query
from .util import approx_tokens
from .worker import MemoryWorker


def make_handler(config: Config, worker: MemoryWorker | None = None):
    worker_ref: dict[str, MemoryWorker] = {}
    worker_lock = Lock()

    def get_worker() -> MemoryWorker:
        # lazy so tests can build a handler without a live worker
        with worker_lock:
            if worker_ref.get("w") is None:
                worker_ref["w"] = worker or MemoryWorker(config)
            return worker_ref["w"]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep stdout quiet
            return

        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: dict) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"))

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def _local_request(self):
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            host = self.headers.get("Host", "")
            origin = self.headers.get("Origin")
            if host not in allowed or (origin and origin != "http://" + host):
                self._json(403, {"error": "仅允许本机页面访问。"})
                return False
            return True

        def do_GET(self):  # noqa: N802
            if not self._local_request():
                return
            path = urlparse(self.path).path
            if path == "/":
                self._send(200, Path(__file__).with_name("trial.html").read_bytes(), "text/html; charset=utf-8")
                return
            conn = db.connect(config.db_path)
            db.init_db(conn)
            try:
                if path == "/health":
                    asm = assemble_memory(config, "prod")
                    self._json(200, {
                        "ok": True,
                        "idea": ideastore.stats(config, conn, "prod"),
                        "memory": {
                            "assembly": asm["assembly"],
                            "injected_chars": len(asm["text"]),
                            "injected_token_estimate": approx_tokens(asm["text"]),
                            "stable_layer": {
                                "present": asm["stable_present"],
                                "path": asm["stable_path"],
                                "chars": len(asm["stable"]) if asm["stable"] else 0,
                                "sha256_prefix": hashlib.sha256(asm["stable"].encode()).hexdigest()[:12]
                                if asm["stable"] else None,
                            },
                            "event_summary": {
                                "path": str(ideastore.idea_path(config, "prod")),
                                "chars": len(asm["event"]),
                                "sha256_prefix": hashlib.sha256(asm["event"].encode()).hexdigest()[:12],
                            },
                        },
                        "tasks": get_worker().counts("prod"),
                        "models": {
                            "main_chat_id": config.main_chat_model,
                            "main_chat_role": "主聊天（含工具循环与最终恢复轮）",
                            "retrieval_id": config.chat_model,
                            "retrieval_role": "独立记忆检索 / 后台整理（不随主聊天切换）",
                        },
                        "retrieval_agent": {
                            "version": AGENT_VERSION,
                            "engine": "bm25 + independent Memory Agent (search/read/expand)",
                            "sources": ["export messages", "chat_log"],
                            "now_injected": "Asia/Shanghai (+08:00)",
                            "time_guarded": "message time strictly before now; selected branch; non-reasoning",
                        },
                    })
                elif path == "/idea":
                    self._send(200, assemble_memory(config, "prod")["text"].encode("utf-8"),
                               "text/markdown; charset=utf-8")
                elif path == "/tasks":
                    w = get_worker()
                    instance = (urlparse(self.path).query or "").replace("instance=", "") or "prod"
                    self._json(200, {"counts": w.counts(instance), "tasks": w.list_tasks(instance)})
                else:
                    self._json(404, {"error": "not found"})
            finally:
                conn.close()

        def do_POST(self):  # noqa: N802
            if not self._local_request():
                return
            if self.headers.get_content_type() != "application/json":
                self._json(415, {"error": "需要 JSON 请求。"})
                return
            path = urlparse(self.path).path
            try:
                if not 0 < int(self.headers.get("Content-Length", 0)) <= 2_000_000:
                    raise ValueError()
                body = self._read_json()
                if not isinstance(body, dict) or body.get("instance", "prod") not in ("prod", "test"):
                    raise ValueError()
                if path in ("/query", "/chat", "/search"):
                    field = "question" if path == "/query" else ("query" if path == "/search" else "message")
                    message = body.get(field, "")
                    if not isinstance(message, str) or not message.strip():
                        raise ValueError()
            except (ValueError, TypeError):
                self._json(400, {"error": "请求格式不正确。"})
                return

            instance = body.get("instance", "prod")
            conn = db.connect(config.db_path)
            try:
                db.init_db(conn)
                if path == "/query":
                    result = run_query(config, conn, message, instance=instance, include_history=bool(body.get("history", False)))
                elif path == "/chat":
                    session = body.get("conversation_id")
                    if session is not None and (not isinstance(session, str) or not 1 <= len(session) <= 100):
                        raise ValueError("Invalid session")
                    recent = body.get("recent_limit")
                    result = chat_agentic(
                        config, conn, message, instance=instance, conversation_id=session,
                        recent_limit=int(recent) if recent is not None else None,
                        from_chatlog=bool(body.get("from_chatlog", True)),
                        worker=get_worker(),
                    )
                elif path == "/search":
                    result = memory_agent_search(config, conn, message, instance=instance)
                elif path == "/update":
                    self._json(409, {"error": "本试用入口固定摘要，不执行记忆更新。"})
                    return
                else:
                    self._json(404, {"error": "not found"})
                    return
                self._json(200, result)
            except BudgetExceeded:
                self._json(429, {"error": "已达到本轮试用额度或上下文上限。记忆未因本次失败而更新；可在配置中提高试用额度。"})
            except MissingCredential:
                self._json(503, {"error": "模型凭据未加载，请通过试用启动器重新启动。"})
            except (ValueError, TypeError):
                self._json(400, {"error": "输入格式不正确，请检查后重试。"})
            except EmptyModelResponse:
                # The model answered with empty/truncated/leaked tool markup on both the
                # normal rounds and the recovery turn. The bad answer was discarded and
                # never written to history, so a plain retry is safe.
                self._json(502, {"error": "模型这次没有返回可用的正文（可能是内部工具格式或截断），已丢弃未保存。请重试。"})
            except Exception:
                self._json(502, {"error": "这次请求未完成，可能是模型响应超时或本地处理失败。请重试；若持续发生，请回到 Codex 排查。"})
            finally:
                conn.close()

    return Handler


def serve(config: Config, host: str = "127.0.0.1", port: int = 8787, start_worker: bool = True) -> ThreadingHTTPServer:
    worker = MemoryWorker(config)
    if start_worker:
        worker.start()
    server = ThreadingHTTPServer((host, port), make_handler(config, worker))
    server.memory_worker = worker  # type: ignore[attr-defined]
    return server
