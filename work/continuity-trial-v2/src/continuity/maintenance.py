"""Build maintenance batches and invoke the existing maintenance CLI.

The engine is treated as a fixed external dependency:

    python3 run_update.py --state-dir <dir> --batch <json> --output <json>

It returns ``status`` in {ok, noop, duplicate, error, rejected_*, conflict,
committed_unpublished} plus ``version`` and the published ``summary_path`` /
``memory_path`` / ``evidence_path``. We only *call* it; we never edit it.

``--budget`` on the engine is a CUMULATIVE cap over its ledger (it persists
``state/ledger/usage.json``), so we pass the TOTAL trial budget, never a
per-invocation number, and we never reset the ledger.

``--output-tokens`` is passed on every call so a long consolidation turn is not
truncated by the CLI's small 16k default (the same 64k headroom the personal
bootstrap uses). This is a plain argv value: it is never a credential and never a
per-invocation budget.

The subprocess inherits the current environment, so when the trial is launched
through the trusted credential wrapper the DeepSeek key is already present and is
never written to a file, argv, or log here.
"""

from __future__ import annotations

import json
import os
import re
import selectors
import time
import subprocess
import sys
from pathlib import Path
from typing import Any, Protocol

from .events import current
from .config import Config
from .retrieval import now_shanghai

_SAFE = re.compile(r"[^A-Za-z0-9._-]")

# Engine statuses that mean "the batch left the published memory in a good state".
OK_STATUSES = {"ok", "noop", "duplicate"}


class MaintenanceError(RuntimeError):
    """The maintenance CLI could not be run or produced no usable result."""


def is_budget_error(text: str | None) -> bool:
    """True if an engine message reports the cumulative request budget exhausted."""
    low = (text or "").lower()
    return "budget" in low or "预算" in low


def read_ledger_usage(state_dir: Path | str) -> dict[str, int]:
    """Read the engine's cumulative ledger usage (no reset, no side effects)."""
    path = Path(state_dir) / "ledger" / "usage.json"
    if not path.exists():
        return {"requests": 0, "input_tokens": 0, "output_tokens": 0, "errors": 0}
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"requests": 0, "input_tokens": 0, "output_tokens": 0, "errors": 0}
    return {
        "requests": int(data.get("requests") or 0),
        "input_tokens": int(data.get("input_tokens") or 0),
        "output_tokens": int(data.get("output_tokens") or 0),
        "errors": int(data.get("errors") or 0),
    }


class MaintenanceRunner(Protocol):
    def run(self, batch: dict[str, Any], work_dir: Path) -> dict[str, Any]:
        ...


def _iso(epoch: float) -> str:
    return now_shanghai(float(epoch))


def build_cli_argv(config: Config, batch_path: Path, out_path: Path) -> list[str]:
    """The exact argv handed to the maintenance CLI (pure, no side effects).

    Kept as a separate function so the argument contract (state dir, output path,
    model, cumulative budget, output headroom) can be asserted without running a
    subprocess.
    """
    return [
        sys.executable, str(config.maint_cli),
        "--state-dir", str(config.maint_state_dir),
        "--batch", str(batch_path),
        "--output", str(out_path),
        "--model", config.maint_model,
        "--budget", str(config.maint_total_budget),
        "--output-tokens", str(config.maint_max_output_tokens),
    ]


def build_turn_batch(conn, config: Config, user_row_id: int, assistant_row_id: int) -> dict[str, Any]:
    """Assemble the exact batch for one successful chat turn.

    The raw user and assistant text is read back from the trial's own ``chat_log``
    (never re-typed), so a queue retry resends byte-identical content and the
    engine's fingerprint/idempotency check makes the retry a no-op instead of a
    duplicate message.
    """
    rows = conn.execute(
        "SELECT id, created_at, conversation_id, role, content FROM chat_log WHERE id IN (?,?)",
        (user_row_id, assistant_row_id),
    ).fetchall()
    by_id = {r["id"]: r for r in rows}
    if user_row_id not in by_id or assistant_row_id not in by_id:
        raise MaintenanceError("cannot build batch: chat rows not found")
    ordered = [by_id[user_row_id], by_id[assistant_row_id]]
    messages = []
    for r in ordered:
        if r["role"] not in ("user", "assistant"):
            raise MaintenanceError(f"unexpected chat role: {r['role']}")
        messages.append({
            "id": f"chat#{r['id']}",
            "conversation_id": r["conversation_id"],
            "role": r["role"],
            "message_at": _iso(r["created_at"]),
            "text": r["content"] or "",
        })
    batch_id = f"turn-chat{user_row_id}-chat{assistant_row_id}"
    return {
        "user_id": config.user_id,
        "batch_id": batch_id,
        "now": now_shanghai(),
        "messages": messages,
    }


class CliMaintenanceRunner:
    """Default runner: invoke ``run_update.py`` as a subprocess."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def run(self, batch: dict[str, Any], work_dir: Path) -> dict[str, Any]:
        cli = Path(self.config.maint_cli)
        if not cli.exists():
            raise MaintenanceError(f"maintenance CLI not found: {cli}")
        work_dir.mkdir(parents=True, exist_ok=True)
        safe = _SAFE.sub("_", str(batch.get("batch_id") or "batch"))[:60] or "batch"
        batch_path = work_dir / f"{safe}.batch.json"
        out_path = work_dir / f"{safe}.out.json"
        batch_path.write_text(json.dumps(batch, ensure_ascii=False, indent=2), "utf-8")
        if out_path.exists():
            out_path.unlink()
        cmd = build_cli_argv(self.config, batch_path, out_path)
        observer = current()
        if observer:
            return self._run_observed(cmd, out_path, observer)
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=self.config.maint_timeout_seconds, env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired as exc:  # pragma: no cover - timing dependent
            raise MaintenanceError(f"maintenance CLI timed out after {self.config.maint_timeout_seconds}s") from exc
        if not out_path.exists():
            detail = (proc.stderr or proc.stdout or "").strip()[-400:]
            raise MaintenanceError(f"CLI produced no output file (rc={proc.returncode}): {detail}")
        try:
            return json.loads(out_path.read_text("utf-8"))
        except json.JSONDecodeError as exc:
            raise MaintenanceError(f"CLI output is not valid JSON: {exc}") from exc

    def _run_observed(self, cmd, out_path, observer):
        read_fd, write_fd = os.pipe()
        env = os.environ.copy()
        env['CONTINUITY_OBSERVER_FD'] = str(write_fd)
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        proc = None
        started = time.monotonic()
        pending = b''
        saw_ready = False
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    env=env, pass_fds=(write_fd,))
            os.close(write_fd)
            write_fd = None
            with selectors.DefaultSelector() as selector:
                selector.register(read_fd, selectors.EVENT_READ)
                while True:
                    if time.monotonic() - started > self.config.maint_timeout_seconds:
                        raise MaintenanceError('maintenance CLI timed out')
                    if not selector.select(timeout=0.2):
                        if proc.poll() is not None:
                            break
                        continue
                    chunk = os.read(read_fd, 65536)
                    if not chunk:
                        break
                    pending += chunk
                    while b'\n' in pending:
                        line, pending = pending.split(b'\n', 1)
                        try:
                            event = json.loads(line)
                            phase = event['phase']
                            if phase.startswith('maintenance_') and isinstance(event.get('data'), dict):
                                saw_ready |= phase == 'maintenance_observer_ready'
                                observer.emit(phase, **event['data'], observed_at=event.get('observed_at'))
                        except (ValueError, KeyError, TypeError, AttributeError):
                            observer.emit('maintenance_observer_gap', reason='invalid JSONL')
                    if len(pending) > 256 * 1024:
                        raise MaintenanceError('observer record exceeds limit')
            remaining = max(0.01, self.config.maint_timeout_seconds - (time.monotonic() - started))
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired as exc:
                raise MaintenanceError('maintenance CLI timed out') from exc
            if not saw_ready:
                observer.emit('maintenance_observer_gap', reason='CLI has no live observer; only process/result observed')
            if not out_path.exists():
                raise MaintenanceError('maintenance CLI produced no output file')
            return json.loads(out_path.read_text('utf-8'))
        finally:
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait()
            os.close(read_fd)
            if write_fd is not None:
                os.close(write_fd)
