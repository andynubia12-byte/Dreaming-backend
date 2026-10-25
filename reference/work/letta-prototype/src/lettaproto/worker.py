"""Independent background Memory Worker.

Responsibilities:
  * persist and process memory-update suggestions (submitted -> processing -> saved|failed);
  * idle-incrementally process NEW chat messages only (no full-history rebuild);
  * serialize memory writes; be idempotent and recover unfinished tasks after restart;
  * never let a failure corrupt the previous working memory.

Retrieval (search) is served separately and synchronously from agents.memory_agent_search,
so consolidation never blocks an in-chat search.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

from . import agents, db
from .config import Config
from .deepseek import BudgetExceeded
from .retrieval import now_shanghai

TASK_MARKER = "<!-- task:{tid} -->"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cursor_key(instance: str) -> str:
    return f"update_cursor:{instance}"


class MemoryWorker:
    def __init__(self, config: Config):
        self.config = config
        self.write_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: str | None = None
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = db.connect(self.config.db_path)
        db.init_db(conn)
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        conn.close()

    def recover(self) -> int:
        conn = self._connect()
        try:
            cur = conn.execute(
                "UPDATE memory_tasks SET status='submitted', updated_at=?, error=COALESCE(error,'')||'[recovered]'"
                " WHERE status='processing'",
                (_now(),),
            )
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()

    def enqueue(
        self,
        instance: str,
        kind: str,
        spec: dict[str, Any] | None = None,
        source_ids: list[Any] | None = None,
        dedup_key: str | None = None,
    ) -> dict[str, Any]:
        conn = self._connect()
        try:
            if dedup_key:
                row = conn.execute("SELECT id, status FROM memory_tasks WHERE dedup_key=?", (dedup_key,)).fetchone()
                if row:
                    return {"task_id": row["id"], "status": row["status"], "deduplicated": True}
            cur = conn.execute(
                "INSERT INTO memory_tasks(dedup_key, created_at, updated_at, instance, kind, status, spec_json, source_ids_json, attempts)"
                " VALUES(?,?,?,?,?,?,?,?,0)",
                (dedup_key, _now(), _now(), instance, kind, "submitted",
                 json.dumps(spec or {}, ensure_ascii=False), json.dumps(source_ids or [], ensure_ascii=False)),
            )
            conn.commit()
            return {"task_id": cur.lastrowid, "status": "submitted", "deduplicated": False}
        finally:
            conn.close()

    def counts(self, instance: str | None = None) -> dict[str, int]:
        conn = self._connect()
        try:
            sql = "SELECT status, COUNT(*) c FROM memory_tasks"
            args: tuple = ()
            if instance:
                sql += " WHERE instance=?"
                args = (instance,)
            rows = conn.execute(sql + " GROUP BY status", args).fetchall()
            out = {"submitted": 0, "processing": 0, "saved": 0, "failed": 0}
            for r in rows:
                out[r["status"]] = r["c"]
            return out
        finally:
            conn.close()

    def list_tasks(self, instance: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            sql = "SELECT * FROM memory_tasks"
            args: list[Any] = []
            if instance:
                sql += " WHERE instance=?"
                args.append(instance)
            sql += " ORDER BY id DESC LIMIT ?"
            args.append(limit)
            out = []
            for r in conn.execute(sql, args).fetchall():
                out.append({
                    "id": r["id"], "kind": r["kind"], "status": r["status"], "instance": r["instance"],
                    "created_at": r["created_at"], "updated_at": r["updated_at"], "attempts": r["attempts"],
                    "source_ids": json.loads(r["source_ids_json"] or "[]"),
                    "result": json.loads(r["result_json"] or "null"),
                    "error": r["error"],
                })
            return out
        finally:
            conn.close()

    def _claim_next(self) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM memory_tasks WHERE status='submitted' ORDER BY id LIMIT 1"
            ).fetchone()
            if not row:
                return None
            conn.execute(
                "UPDATE memory_tasks SET status='processing', updated_at=?, started_at=? WHERE id=?",
                (_now(), _now(), row["id"]),
            )
            conn.commit()
            return dict(row)
        finally:
            conn.close()

    def _finish(self, task_id: int, status: str, result: dict | None, error: str | None, attempts: int) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE memory_tasks SET status=?, updated_at=?, finished_at=?, result_json=?, error=?, attempts=? WHERE id=?",
                (status, _now(), _now(), json.dumps(result or {}, ensure_ascii=False), error, attempts, task_id),
            )
            conn.commit()
        finally:
            conn.close()

    def maybe_enqueue_idle(self, instance: str = "prod") -> dict[str, Any] | None:
        if not self.config.worker_idle_enabled:
            return None
        conn = self._connect()
        try:
            cursor = float(db.get_meta(conn, _cursor_key(instance), "0") or 0)
            row = conn.execute(
                "SELECT MAX(created_at) mx, COUNT(*) c FROM chat_log WHERE instance=? AND role='user' AND created_at > ?",
                (instance, cursor),
            ).fetchone()
            if not row["mx"] or not row["c"]:
                return None
            return self.enqueue(instance, "idle", {"trigger": "idle"}, [], dedup_key=f"idle:{instance}:{row['mx']}")
        finally:
            conn.close()

    def process_task(self, task: dict[str, Any]) -> dict[str, Any]:
        conn = self._connect()
        try:
            return self._process_task(conn, task)
        finally:
            conn.close()

    def _process_task(self, conn: sqlite3.Connection, task: dict[str, Any]) -> dict[str, Any]:
        instance = task["instance"]
        spec = json.loads(task["spec_json"] or "{}")
        source_ids = json.loads(task["source_ids_json"] or "[]")

        if task["kind"] == "idle":
            cursor = float(db.get_meta(conn, _cursor_key(instance), "0") or 0)
            rows = conn.execute(
                "SELECT id, created_at, content FROM chat_log WHERE instance=? AND role='user' AND created_at > ? ORDER BY created_at",
                (instance, cursor),
            ).fetchall()
            if not rows:
                return {"status": "noop", "reason": "no new messages"}
            evidence = [{"id": f"chat#{r['id']}", "role": "user", "time": None, "text": r["content"]} for r in rows]
            max_ts = max(r["created_at"] for r in rows)
            statement = None
            trigger = "idle"
        else:
            verified, rejected = agents.verify_source_ids(conn, source_ids, instance)
            if not verified:
                return {"status": "failed", "error": f"no verifiable USER source (rejected={rejected})"}
            evidence = verified
            max_ts = None
            statement = spec.get("statement")
            trigger = task["kind"]

        current = agents.ideastore.load(self.config, instance)
        if TASK_MARKER.format(tid=task["id"]) in current:
            if max_ts:
                db.set_meta(conn, _cursor_key(instance), str(max_ts))
                conn.commit()
            return {"status": "saved", "note": "already applied (idempotent)"}

        with self.write_lock:
            try:
                info = agents.memory_agent_consolidate(
                    self.config, conn, instance=instance, evidence=evidence, statement=statement,
                    note=f"memory agent {trigger}", trigger=trigger, now=now_shanghai(),
                )
            except BudgetExceeded as exc:
                return {"status": "failed", "error": f"budget: {exc}"}
            if info.get("status") == "noop":
                if max_ts:
                    db.set_meta(conn, _cursor_key(instance), str(max_ts))
                    conn.commit()
                return {"status": "saved", "note": "nothing durable", "result": info}
            text = agents.ideastore.load(self.config, instance)
            marker = TASK_MARKER.format(tid=task["id"])
            if marker not in text:
                agents.ideastore.save(self.config, conn, text.rstrip() + "\n" + marker + "\n",
                                      note="task marker", instance=instance)

        if max_ts:
            db.set_meta(conn, _cursor_key(instance), str(max_ts))
            conn.commit()
        return {"status": "saved", "result": info}

    def run_once(self, instance: str | None = None) -> dict[str, Any] | None:
        task = self._claim_next()
        if task is None:
            return None
        tid = task["id"]
        attempts = int(task["attempts"] or 0) + 1
        try:
            result = self.process_task(task)
            status = result.get("status", "saved")
            if status in ("saved", "noop"):
                self._finish(tid, "saved", result, None, attempts)
            else:
                self._finish(tid, "failed", result, result.get("error"), attempts)
            return {"task_id": tid, "status": status, "result": result}
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, BudgetExceeded) or attempts >= self.config.worker_max_attempts:
                self._finish(tid, "failed", None, str(exc)[:300], attempts)
                return {"task_id": tid, "status": "failed", "error": str(exc)[:300]}
            self._finish(tid, "submitted", None, str(exc)[:300], attempts)
            return {"task_id": tid, "status": "retry", "error": str(exc)[:200]}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.recover()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="memory-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                if self.run_once() is None:
                    self.maybe_enqueue_idle()
                    if self.run_once() is None:
                        self._stop.wait(self.config.worker_poll_seconds)
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"{type(exc).__name__}: {exc}"
                self._stop.wait(self.config.worker_poll_seconds)
