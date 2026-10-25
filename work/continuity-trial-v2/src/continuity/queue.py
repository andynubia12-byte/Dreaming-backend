"""Durable, strictly-serial maintenance queue.

One row per successful chat turn. A background thread claims tasks oldest-first
and runs them one at a time (never concurrently), so:

* the maintenance engine sees a monotonically increasing ``now`` (it rejects a
  ``now`` earlier than the last accepted one), and
* the OLDEST unfinished turn blocks every later turn: a failure (retryable or
  exhausted) is never leapfrogged, so a later turn can never advance the engine's
  ``now`` past an unresolved earlier batch. The batch ``now`` is never rewritten.

Reliability properties required by the trial:

* **No lost turns** — every successful turn is persisted and enqueued; the queue
  row and payload live in SQLite. Persistence + enqueue happen in ONE SQLite
  transaction (see ``chat.chat_agentic``), so a crash cannot leave a saved answer
  without its queue row.
* **Restart recovery** — rows stuck in ``processing`` are reset to ``pending`` at
  startup and re-run; successful turns missing a queue row are re-enqueued
  (``reconcile_turns``).
* **Idempotent retries** — the batch payload is frozen at enqueue time (raw text
  read back from ``chat_log``), so re-running the same row resends identical
  content and the engine dedups it instead of duplicating messages. The queue
  never rewrites ``now`` and never turns a ``conflict`` into a success; only the
  engine's own ``duplicate`` verdict counts.
* **Old summary preserved on failure** — a failed run never publishes, so the
  previously published summary stays the one that chat reads.
* **Honest budget** — the engine's ``--budget`` is a cumulative ledger cap; the
  queue reports remaining budget and stops (blocking, not hammering) when it is
  exhausted, never resetting or growing the ledger.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from . import db
from .events import EventHub, Observer, observe
from .config import Config
from .maintenance import (
    OK_STATUSES,
    MaintenanceRunner,
    build_turn_batch,
    is_budget_error,
    read_ledger_usage,
)

STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_DONE = "done"
STATUS_FAILED = "failed"


def _now() -> float:
    return time.time()


class MaintenanceQueue:
    def __init__(
        self,
        config: Config,
        runner: MaintenanceRunner,
        *,
        poll_seconds: float | None = None,
        max_attempts: int | None = None,
    ) -> None:
        self.config = config
        self.runner = runner
        self.poll_seconds = poll_seconds if poll_seconds is not None else config.queue_poll_seconds
        self.max_attempts = max_attempts if max_attempts is not None else config.queue_max_attempts
        self.runs_dir = config.runs_dir / "maint"
        self.last_error: str | None = None
        self.budget_exhausted = False
        self.events = EventHub()
        self._origins = {}
        self._origins_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._init_db()

    def _turn_committed(self, queued, observer, session_id):
        ids = observer.ids if observer else {}
        self._origins[queued['task_id']] = {
            'session_id': session_id, 'request_id': ids.get('request_id'),
            'parent_id': ids.get('run_id')}
        Observer(self.events, session_id=session_id, request_id=ids.get('request_id'),
                 parent_id=ids.get('run_id'), task_id=queued['task_id'],
                 channel='maintenance').emit('maintenance_queued', **queued)

    def turn_committed(self, queued, observer, session_id):
        with self._origins_lock:
            self._turn_committed(queued, observer, session_id)

    def commit_turn(self, conn, queued, observer, session_id):
        # A worker may claim immediately after COMMIT, but cannot publish its
        # processing event before the origin + committed queue event are visible.
        with self._origins_lock:
            conn.commit()
            self._turn_committed(queued, observer, session_id)

    # ---- connections -----------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = db.connect(self.config.db_path)
        db.init_db(conn)
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        conn.close()

    # ---- recovery --------------------------------------------------------
    def recover(self) -> int:
        """Reset interrupted (processing) rows to pending. Called on startup."""
        conn = self._connect()
        try:
            cur = conn.execute(
                "UPDATE maint_queue SET status=?, updated_at=? WHERE status=?",
                (STATUS_PENDING, _now(), STATUS_PROCESSING),
            )
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()

    def reconcile_turns(self) -> int:
        """Re-enqueue successful turns that have no queue row (crash safety net).

        Persistence + enqueue are atomic, so this normally finds nothing; it exists
        so an interrupted write (or data from an older build) is repaired on the
        next start. Pairs, within each session, a user message with the assistant
        reply that immediately follows it (sessions are serialized, so the pairing
        is unambiguous). Idempotent: ``batch_id`` is unique and content-stable.
        """
        conn = self._connect()
        try:
            existing = {r["batch_id"] for r in conn.execute("SELECT batch_id FROM maint_queue")}
            rows = conn.execute(
                "SELECT id, conversation_id, role FROM chat_log WHERE instance=?"
                " ORDER BY conversation_id, id",
                (self.config.instance,),
            ).fetchall()
            pending_user: int | None = None
            pending_conv: str | None = None
            added = 0
            for r in rows:
                if r["role"] == "user":
                    pending_user, pending_conv = r["id"], r["conversation_id"]
                elif r["role"] == "assistant" and pending_user is not None and pending_conv == r["conversation_id"]:
                    batch_id = f"turn-chat{pending_user}-chat{r['id']}"
                    if batch_id not in existing:
                        batch = build_turn_batch(conn, self.config, pending_user, r["id"])
                        self._insert_batch(conn, batch)
                        existing.add(batch_id)
                        added += 1
                    pending_user = None
            if added:
                conn.commit()
            return added
        finally:
            conn.close()

    # ---- enqueue ---------------------------------------------------------
    def enqueue_batch(self, batch: dict[str, Any], *, kind: str = "turn") -> dict[str, Any]:
        conn = self._connect()
        try:
            result = self._insert_batch(conn, batch, kind=kind)
            conn.commit()
            return result
        finally:
            conn.close()

    @staticmethod
    def _insert_batch(conn: sqlite3.Connection, batch: dict[str, Any], *, kind: str = "turn") -> dict[str, Any]:
        """Insert a queue row on the caller's connection (NO commit)."""
        batch_id = str(batch.get("batch_id") or "")
        existing = conn.execute(
            "SELECT id, status FROM maint_queue WHERE batch_id=?", (batch_id,)
        ).fetchone()
        if existing:
            return {"task_id": existing["id"], "batch_id": batch_id,
                    "status": existing["status"], "deduplicated": True}
        cur = conn.execute(
            "INSERT INTO maint_queue(batch_id, kind, status, created_at, updated_at, attempts,"
            " user_id, payload_json) VALUES(?,?,?,?,?,0,?,?)",
            (batch_id, kind, STATUS_PENDING, _now(), _now(), batch.get("user_id"),
             json.dumps(batch, ensure_ascii=False)),
        )
        return {"task_id": cur.lastrowid, "batch_id": batch_id,
                "status": STATUS_PENDING, "deduplicated": False}

    def enqueue_turn(
        self, user_row_id: int, assistant_row_id: int, *, kind: str = "turn",
        conn: sqlite3.Connection | None = None,
    ) -> dict[str, Any]:
        """Enqueue a turn. With ``conn`` the row is written on the caller's
        connection WITHOUT committing, so the caller can commit it in the same
        transaction as the assistant message (atomic persist + enqueue)."""
        own = conn is None
        if own:
            conn = self._connect()
        try:
            batch = build_turn_batch(conn, self.config, user_row_id, assistant_row_id)
            result = self._insert_batch(conn, batch, kind=kind)
            if own:
                conn.commit()
            return result
        finally:
            if own:
                conn.close()
        return self.enqueue_batch(batch, kind=kind)

    # ---- inspection ------------------------------------------------------
    def counts(self) -> dict[str, int]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT status, COUNT(*) c FROM maint_queue GROUP BY status").fetchall()
            out = {STATUS_PENDING: 0, STATUS_PROCESSING: 0, STATUS_DONE: 0, STATUS_FAILED: 0}
            for r in rows:
                out[r["status"]] = r["c"]
            return out
        finally:
            conn.close()

    def list_tasks(self, limit: int = 20) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM maint_queue ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [self._row_view(r) for r in rows]
        finally:
            conn.close()

    def get_task(self, task_id: int) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM maint_queue WHERE id=?", (task_id,)).fetchone()
            return self._row_view(row) if row else None
        finally:
            conn.close()

    def last_task(self) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM maint_queue ORDER BY id DESC LIMIT 1").fetchone()
            return self._row_view(row) if row else None
        finally:
            conn.close()

    def head_task(self) -> dict[str, Any] | None:
        """The OLDEST unfinished task (pending/failed/processing), or None."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM maint_queue WHERE status IN (?,?,?) ORDER BY id LIMIT 1",
                (STATUS_PENDING, STATUS_FAILED, STATUS_PROCESSING),
            ).fetchone()
            return self._row_view(row) if row else None
        finally:
            conn.close()

    def blocked_task(self) -> dict[str, Any] | None:
        """The task currently blocking ALL later work, or None.

        Under strict FIFO an exhausted (attempts>=max) failed head stops the queue
        until an explicit retry; this is the recovery target the UI must show.
        """
        head = self.head_task()
        if head and head["status"] == STATUS_FAILED and int(head["attempts"] or 0) >= self.max_attempts:
            return head
        return None

    @staticmethod
    def _row_view(row: sqlite3.Row) -> dict[str, Any]:
        try:
            result = json.loads(row["result_json"] or "null")
        except json.JSONDecodeError:
            result = None
        return {
            "id": row["id"], "batch_id": row["batch_id"], "kind": row["kind"],
            "status": row["status"], "attempts": row["attempts"],
            "created_at": row["created_at"], "updated_at": row["updated_at"],
            "finished_at": row["finished_at"], "summary_version": row["summary_version"],
            "error": row["error"], "result": result,
            "retryable": row["status"] == STATUS_FAILED,
        }

    # ---- manual retry ----------------------------------------------------
    def retry(self, task_id: int) -> dict[str, Any]:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM maint_queue WHERE id=?", (task_id,)).fetchone()
            if not row:
                raise KeyError(f"task {task_id} not found")
            if row["status"] == STATUS_DONE:
                return {"task_id": task_id, "status": STATUS_DONE, "note": "already done"}
            if row["status"] == STATUS_PROCESSING:
                # Never re-run a task that a worker is currently processing.
                return {"task_id": task_id, "status": STATUS_PROCESSING,
                        "note": "正在处理中，已忽略重复重试"}
            conn.execute(
                "UPDATE maint_queue SET status=?, attempts=0, error=NULL, updated_at=? WHERE id=?",
                (STATUS_PENDING, _now(), task_id),
            )
            conn.commit()
            return {"task_id": task_id, "status": STATUS_PENDING}
        finally:
            conn.close()

    # ---- processing ------------------------------------------------------
    def _claim_next(self) -> dict[str, Any] | None:
        """Claim the OLDEST unfinished task, with strict head-of-line blocking.

        The oldest row among {pending, failed} is the head. If it is a failure
        whose attempts are exhausted, nothing is claimed: every later turn stays
        blocked until an explicit ``retry`` resets that head. This keeps the
        engine's ``now`` strictly increasing and never rewrites a batch ``now``.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM maint_queue WHERE status IN (?,?,?) ORDER BY id LIMIT 1",
                (STATUS_PENDING, STATUS_FAILED, STATUS_PROCESSING),
            ).fetchone()
            if not row:
                return None
            if row["status"] == STATUS_PROCESSING:
                return None  # a stuck/active processing head blocks everything else
            if row["status"] == STATUS_FAILED and int(row["attempts"] or 0) >= self.max_attempts:
                return None  # terminal failure at the head -> block all later turns
            cur = conn.execute(
                "UPDATE maint_queue SET status=?, updated_at=?, started_at=?"
                " WHERE id=? AND status=?",
                (STATUS_PROCESSING, _now(), _now(), row["id"], row["status"]),
            )
            if cur.rowcount != 1:
                conn.rollback()
                return None
            conn.commit()
            return dict(row)
        finally:
            conn.close()

    def _finish(self, task_id: int, status: str, *, attempts: int, result: dict | None,
                error: str | None, summary_version: int | None) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE maint_queue SET status=?, updated_at=?, finished_at=?, attempts=?,"
                " result_json=?, error=?, summary_version=? WHERE id=?",
                (status, _now(), _now(), attempts,
                 json.dumps(result or {}, ensure_ascii=False), error, summary_version, task_id),
            )
            conn.commit()
        finally:
            conn.close()

    def budget_status(self) -> dict[str, Any]:
        """Honest view of the cumulative maintenance budget (no reset)."""
        usage = read_ledger_usage(self.config.maint_state_dir)
        total = int(self.config.maint_total_budget)
        used = int(usage["requests"])
        remaining = max(0, total - used)
        return {
            "total_requests": total,
            "used_requests": used,
            "remaining_requests": remaining,
            "ledger": str(self.config.maint_state_dir / "ledger" / "usage.json"),
            "exhausted": self.budget_exhausted or used >= total,
            "note": "engine --budget is a cumulative ledger cap; never auto-reset or auto-increased",
        }

    def run_once(self) -> dict[str, Any] | None:
        task = self._claim_next()
        if task is None:
            return None
        task_id = task["id"]
        attempts = int(task["attempts"] or 0) + 1
        try:
            batch = json.loads(task["payload_json"] or "{}")
        except json.JSONDecodeError:
            self._finish(task_id, STATUS_FAILED, attempts=attempts, result=None,
                         error="payload is not valid JSON", summary_version=None)
            return {"task_id": task_id, "status": STATUS_FAILED, "error": "bad payload"}
        with self._origins_lock:
            origin = self._origins.get(task_id, {})
        session = origin.get('session_id') or next(
            (m.get('conversation_id') for m in batch.get('messages', []) if m.get('conversation_id')), '')
        observer = Observer(self.events, session_id=session,
                            request_id=origin.get('request_id') or task['batch_id'],
                            parent_id=origin.get('parent_id'), task_id=task_id, channel='maintenance')
        observer.emit('maintenance_processing', batch_id=task['batch_id'], attempt=attempts)
        try:
            with observe(observer):
                result = self.runner.run(batch, self.runs_dir)
            observer.emit('maintenance_engine_result', result={k: v for k, v in result.items()
                          if k in ('status', 'version', 'published_version', 'new_messages',
                                   'claims_applied', 'applied', 'expired', 'dropped_ops',
                                   'bounds_backfilled', 'model_calls', 'summary_regenerated',
                                   'repair_triggered', 'repair_applied', 'trace_path')})
            status = result.get("status")
            if status in OK_STATUSES:
                self._finish(task_id, STATUS_DONE, attempts=attempts, result=result, error=None,
                             summary_version=result.get("version"))
                observer.emit('maintenance_done', version=result.get('version'), status=status)
                with self._origins_lock:
                    self._origins.pop(task_id, None)
                return {"task_id": task_id, "status": STATUS_DONE, "result": result}
            error = str(result.get("error") or status or "unknown failure")
            if is_budget_error(error):
                # Cumulative budget is gone: block the head, do not hammer, and
                # require an explicit retry after the budget is raised.
                self.budget_exhausted = True
                attempts = max(attempts, self.max_attempts)
            self._finish(task_id, STATUS_FAILED, attempts=attempts, result=result, error=error,
                         summary_version=result.get("version"))
            observer.emit('maintenance_failed', status=status, budget_exhausted=self.budget_exhausted)
            return {"task_id": task_id, "status": STATUS_FAILED, "error": error, "result": result,
                    "budget_exhausted": self.budget_exhausted}
        except Exception as exc:  # noqa: BLE001 - any runner failure is retried
            self.last_error = f"{type(exc).__name__}: {exc}"
            error = str(exc)[:400]
            if is_budget_error(error):
                self.budget_exhausted = True
                attempts = max(attempts, self.max_attempts)
            self._finish(task_id, STATUS_FAILED, attempts=attempts, result=None,
                         error=error, summary_version=None)
            observer.emit('maintenance_failed', error_type=type(exc).__name__,
                          budget_exhausted=self.budget_exhausted)
            return {"task_id": task_id, "status": STATUS_FAILED, "error": error,
                    "budget_exhausted": self.budget_exhausted}

    # ---- background thread ----------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.recover()
        self.reconcile_turns()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="maint-queue", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                info = self.run_once()
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"{type(exc).__name__}: {exc}"
                info = None
            if info is None:
                self._stop.wait(self.poll_seconds)
            elif info.get("status") == STATUS_DONE:
                continue  # immediately try the next turn
            else:
                # back off between retries so a hard failure is not hammered
                self._stop.wait(self.poll_seconds)
