"""Minimal local HTTP API for the continuity trial (127.0.0.1 only).

Adapted from ``work/generated-memory-demo/src/lettaproto/api.py``
(sha256 32d91210bfb783e1b8bb7859f04cd08c4de647439ca59e6af673efea88c5f0ce).

Endpoints:
  GET  /                         trial page (HTML)
  GET  /health                   models, memory pointer, queue counts
  GET  /memory                   current published summary + version
  GET  /status                   queue phase + counters + last task
  GET  /history?conversation_id= one session's stored turns (refresh recovery)
  GET  /sessions                 known sessions (for a session picker)
  POST /chat                     Pro main chat; persists turn; enqueues maintenance
  POST /search                   on-demand Flash retrieval over the trial's own chats
  POST /retry                    re-enqueue a failed maintenance task

There is no endpoint that writes memory directly: the main chat cannot write
memory, and the old ``/update`` path is gone.
"""

from __future__ import annotations

import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
import time
from urllib.parse import parse_qs, urlparse

from . import db
from .agents import AGENT_VERSION, memory_agent_search
from .chat import EmptyModelResponse, chat_agentic
from .config import Config
from .deepseek import BudgetExceeded, MissingCredential
from .history import history_status
from .events import Observer, observe
from .queue import (
    MaintenanceQueue,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_PROCESSING,
)
from .summary_store import empty_summary_hint, read_published
from .util import approx_tokens, iso_from_epoch, looks_like_tool_serialization


def _session_messages(conn, instance: str, conversation_id: str, limit: int = 200) -> list[dict]:
    rows = conn.execute(
        "SELECT id, created_at, role, content FROM chat_log WHERE instance=? AND conversation_id=?"
        " ORDER BY id ASC LIMIT ?",
        (instance, conversation_id, limit),
    ).fetchall()
    out = []
    for r in rows:
        text = r["content"] or ""
        if r["role"] == "assistant" and (not text.strip() or looks_like_tool_serialization(text)):
            continue
        out.append({
            "id": r["id"], "role": r["role"], "text": text,
            "created_at": r["created_at"], "time": iso_from_epoch(r["created_at"]),
        })
    return out


def make_handler(config: Config, queue: MaintenanceQueue):
    session_locks: dict[str, Lock] = {}
    session_locks_guard = Lock()

    def session_lock(session: str) -> Lock:
        """One lock per session so same-session chats run strictly in order
        (deterministic user/assistant ordering -> unambiguous pairing + no
        double-execution under concurrency)."""
        with session_locks_guard:
            return session_locks.setdefault(session, Lock())

    # Process-lifetime idempotency: no ID eviction/re-execution. At capacity refuse
    # new runs, keeping all prior tombstones; refresh only reconnects GET SSE.
    runs = {}
    runs_guard = Lock()

    def start_run(session, request_id, message):
        digest = hashlib.sha256(message.encode()).hexdigest()
        with runs_guard:
            if request_id in runs:
                run = runs[request_id]
                if run['session_id'] != session or run['digest'] != digest:
                    raise ValueError('request_id conflict')
                return run
            if len(runs) >= 256:
                raise ValueError('run registry full; reconnect existing request or restart explicitly')
            observer = Observer(queue.events, session_id=session, request_id=request_id)
            run = {'session_id': session, 'request_id': request_id, 'digest': digest,
                   'terminal': None}
            runs[request_id] = run

        def work():
            with observe(observer):
                observer.emit('request_accepted')
                try:
                    # This worker owns its connection. SSE never takes the lock.
                    with session_lock(session):
                        conn = db.connect(config.db_path)
                        try:
                            db.init_db(conn)
                            result = chat_agentic(
                                config, conn, message,
                                published=read_published(config.maint_state_dir, config.user_id),
                                instance=config.instance, conversation_id=session,
                                maintenance=queue, stream=True)
                        finally:
                            conn.close()
                    event = observer.emit('answer_complete', result=result)
                except Exception as exc:
                    event = observer.emit('request_failed', error_type=type(exc).__name__,
                        error='请求失败；未保存坏回答或排入维护。已保存的用户消息会保留。')
                with runs_guard:
                    run['terminal'] = event
        Thread(target=work, name='chat-' + request_id[:12], daemon=True).start()
        return run

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

        def _local_request(self) -> bool:
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            host = self.headers.get("Host", "")
            origin = self.headers.get("Origin")
            if host not in allowed or (origin and origin != "http://" + host):
                self._json(403, {"error": "仅允许本机页面访问。"})
                return False
            return True

        def _sse(self, session, *, request_id=None, channel=None, until_done=False):
            hub = queue.events
            query = parse_qs(urlparse(self.path).query)
            cursor = self.headers.get('Last-Event-ID') or (query.get('after') or ['0'])[0]
            old_epoch = None
            try:
                if ':' in cursor:
                    old_epoch, cursor = cursor.rsplit(':', 1)
                after = max(0, int(cursor))
            except ValueError:
                self._json(400, {'error': 'invalid SSE cursor'})
                return
            reset = (old_epoch is not None and old_epoch != hub.epoch) or after > hub.seq
            if reset:
                after = 0
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
            self.send_header('Cache-Control', 'no-cache, no-store')
            self.send_header('X-Accel-Buffering', 'no')
            self.send_header('Connection', 'close')
            self.end_headers()
            self.close_connection = True

            def send(event, seq):
                data = json.dumps(event, ensure_ascii=False)
                self.wfile.write(f'id: {hub.epoch}:{seq}\ndata: {data}\n\n'.encode('utf-8'))
                self.wfile.flush()

            started = time.monotonic()
            try:
                send({'phase': 'connected', 'epoch': hub.epoch, 'session_id': session,
                      'data': {'replay_reset': reset}}, after)
                while True:
                    items, cursor, gap = hub.read(after, session_id=session,
                        request_id=request_id, channel=channel, timeout=10)
                    if gap or reset:
                        send({'phase': 'replay_gap', 'epoch': hub.epoch, 'session_id': session,
                              'data': {'reason': 'bounded replay expired or service restarted',
                                       'recover': 'reload /history and /status; never resubmit automatically'}}, after)
                        reset = False
                    for event in items:
                        send(event, event['seq'])
                        if until_done and event['phase'] in ('answer_complete', 'request_failed'):
                            return
                    after = cursor
                    if channel == 'chat':
                        with runs_guard:
                            run = runs.get(request_id)
                            terminal = run.get('terminal') if run else None
                        if terminal:
                            if not any(e['seq'] == terminal['seq'] for e in items):
                                send(terminal, terminal['seq'])
                            return
                    send({'phase': 'heartbeat', 'epoch': hub.epoch, 'session_id': session,
                          'time': time.time(), 'data': {'connection': 'alive'}}, after)
                    # GET subscribers reconnect periodically; never hold SQLite/session locks.
                    if not until_done and time.monotonic() - started >= 60:
                        return
            except (BrokenPipeError, ConnectionResetError, OSError):
                # Worker continues exactly once. Only its validated answer may commit.
                return

        def _published(self):
            return read_published(config.maint_state_dir, config.user_id)

        # ---- GET ----------------------------------------------------------
        def do_GET(self):  # noqa: N802
            if not self._local_request():
                return
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)
            if path == "/":
                html = Path(__file__).with_name("trial.html").read_bytes()
                self._send(200, html, "text/html; charset=utf-8")
                return
            if path == '/events':
                session = (query.get('conversation_id') or [''])[0]
                request_id = (query.get('request_id') or [None])[0]
                channel = (query.get('channel') or ['maintenance'])[0]
                if not session or len(session) > 100 or channel not in ('chat', 'maintenance'):
                    self._json(400, {'error': 'invalid event filter'})
                    return
                if channel == 'chat':
                    with runs_guard:
                        run = runs.get(request_id)
                        valid = run and run['session_id'] == session
                    if not valid:
                        self._json(404, {'error': 'unknown request; inspect saved history; do not auto-resubmit'})
                        return
                self._sse(session, request_id=request_id, channel=channel)
                return
            conn = db.connect(config.db_path)
            db.init_db(conn)
            try:
                pub = self._published()
                if path == "/health":
                    counts = queue.counts()
                    hist = history_status(config, conn)
                    sources = ["this trial's own chat_log"]
                    if hist.get("imported"):
                        sources.insert(0, "imported history (read-only retrieval)")
                    self._json(200, {
                        "ok": True,
                        "models": {
                            "main_chat_id": config.main_chat_model,
                            "main_chat_role": "主聊天（含工具循环与最终恢复轮）",
                            "retrieval_id": config.chat_model,
                            "retrieval_role": "独立记忆检索（不随主聊天切换）",
                        },
                        "memory": {
                            "present": pub.present,
                            "version": pub.version,
                            "chars": len(pub.display_text),
                            "token_estimate": approx_tokens(pub.display_text),
                            "published_at": pub.published_at,
                            "sha256_prefix": hashlib.sha256(pub.display_text.encode()).hexdigest()[:12],
                            "reason": pub.reason,
                        },
                        "queue": counts,
                        "queue_phase": _phase(counts, queue.budget_status(), queue.blocked_task()),
                        "budget": queue.budget_status(),
                        "retrieval_agent": {
                            "version": AGENT_VERSION,
                            "engine": "bm25 + independent Memory Agent (search/read/expand)",
                            "sources": sources,
                            "history_import": hist,
                            "now_injected": "Asia/Shanghai (+08:00)",
                            "time_guarded": "message time strictly before now; non-reasoning",
                        },
                    })
                elif path == "/memory":
                    self._json(200, {
                        "present": pub.present,
                        "version": pub.version,
                        "chars": len(pub.display_text),
                        "published_at": pub.published_at,
                        "summary_path": pub.summary_path,
                        "reason": pub.reason,
                        "text": pub.display_text,
                        "hint": "" if pub.present else empty_summary_hint(),
                    })
                elif path == "/status":
                    counts = queue.counts()
                    last = queue.last_task()
                    head = queue.head_task()
                    blocked = queue.blocked_task()
                    budget = queue.budget_status()
                    self._json(200, {
                        "memory": {"present": pub.present, "version": pub.version,
                                   "published_at": pub.published_at},
                        "queue": {"counts": counts, "phase": _phase(counts, budget, blocked),
                                  "head": head, "blocked": blocked, "last": last},
                        "budget": budget,
                        "sessions": conn.execute(
                            "SELECT COUNT(DISTINCT conversation_id) c FROM chat_log WHERE instance=?",
                            (config.instance,),
                        ).fetchone()["c"],
                    })
                elif path == "/history":
                    session = (query.get("conversation_id") or [None])[0]
                    if not session:
                        self._json(400, {"error": "缺少 conversation_id。"})
                        return
                    self._json(200, {
                        "conversation_id": session,
                        "memory": {"present": pub.present, "version": pub.version},
                        "messages": _session_messages(conn, config.instance, session),
                    })
                elif path == "/sessions":
                    rows = conn.execute(
                        "SELECT conversation_id, COUNT(*) n, MIN(created_at) first_at, MAX(created_at) last_at"
                        " FROM chat_log WHERE instance=? GROUP BY conversation_id ORDER BY last_at DESC LIMIT 50",
                        (config.instance,),
                    ).fetchall()
                    self._json(200, {"sessions": [
                        {"conversation_id": r["conversation_id"], "messages": r["n"],
                         "first_at": iso_from_epoch(r["first_at"]), "last_at": iso_from_epoch(r["last_at"])}
                        for r in rows
                    ]})
                else:
                    self._json(404, {"error": "not found"})
            finally:
                conn.close()

        # ---- POST ---------------------------------------------------------
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
                if not isinstance(body, dict):
                    raise ValueError()
            except (ValueError, TypeError):
                self._json(400, {"error": "请求格式不正确。"})
                return

            session = body.get("conversation_id")
            if session is not None and (not isinstance(session, str) or not 1 <= len(session) <= 100):
                self._json(400, {"error": "会话标识无效。"})
                return

            if path == '/chat/stream':
                message = body.get('message')
                request_id = body.get('request_id')
                if (not session or not isinstance(message, str) or not message.strip()
                    or not isinstance(request_id, str) or not 1 <= len(request_id) <= 100):
                    self._json(400, {'error': 'message, conversation_id and request_id required'})
                    return
                try:
                    start_run(session, request_id, message)
                except ValueError as exc:
                    self._json(409, {'error': str(exc)})
                    return
                self._sse(session, request_id=request_id, channel='chat', until_done=True)
                return

            conn = db.connect(config.db_path)
            db.init_db(conn)
            try:
                pub = self._published()
                if path == "/chat":
                    message = body.get("message", "")
                    if not isinstance(message, str) or not message.strip():
                        raise ValueError()
                    lock = session_lock(session or "_default")
                    with lock:
                        result = chat_agentic(
                            config, conn, message, published=pub, instance=config.instance,
                            conversation_id=session, maintenance=queue,
                        )
                elif path == "/search":
                    message = body.get("query", "")
                    if not isinstance(message, str) or not message.strip():
                        raise ValueError()
                    result = memory_agent_search(
                        config, conn, pub, message, instance=config.instance,
                        conversation_id=session,
                    )
                elif path == "/retry":
                    task_id = body.get("task_id")
                    if not isinstance(task_id, int):
                        raise ValueError()
                    result = queue.retry(task_id)
                else:
                    self._json(404, {"error": "not found"})
                    return
                self._json(200, result)
            except KeyError as exc:
                self._json(404, {"error": f"未找到任务：{exc}"})
            except BudgetExceeded:
                self._json(429, {"error": "已达到本轮试用额度或上下文上限。记忆未因本次失败而更新。"})
            except MissingCredential:
                self._json(503, {"error": "模型凭据未加载，请通过试用启动器重新启动。"})
            except (ValueError, TypeError):
                self._json(400, {"error": "输入格式不正确，请检查后重试。"})
            except EmptyModelResponse:
                self._json(502, {"error": "模型这次没有返回可用的正文（可能是内部工具格式或截断），未保存、未写入记忆。请重试。"})
            except Exception:
                self._json(502, {"error": "这次请求未完成，可能是模型响应超时或本地处理失败。请重试；若持续发生，请回到 Codex 排查。"})
            finally:
                conn.close()

    return Handler


def _phase(counts: dict[str, int], budget: dict | None = None,
           blocked: dict | None = None) -> str:
    if budget and budget.get("exhausted"):
        return "记忆预算已用尽"
    if blocked:
        # an exhausted failed HEAD stops all later work; surface it over pending
        return "失败可重试"
    if counts.get(STATUS_PENDING) or counts.get(STATUS_PROCESSING):
        return "整理中"
    if counts.get(STATUS_FAILED):
        return "失败可重试"
    if counts.get(STATUS_DONE):
        return "已更新"
    return "空闲"


def serve(config: Config, queue: MaintenanceQueue, host: str = "127.0.0.1",
          port: int = 8794) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(config, queue))
    server.maintenance_queue = queue  # type: ignore[attr-defined]
    return server
