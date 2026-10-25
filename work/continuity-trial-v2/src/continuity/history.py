"""Read-only import of an independent ``retrieval_import.sqlite`` for retrieval.

Input (produced by ``history-bootstrap-v1``): an independent
``retrieval_import.sqlite`` (selected branch, non-reasoning, user/assistant,
verbatim text) plus its ``manifest.json``. This trial NEVER reads the raw export
directory and never reads another demo's database.

Guarantees:
* source sqlite is opened **read-only**; the source is never modified;
* the manifest's REQUIRED counts (conversations / evidence_messages /
  user_messages / assistant_messages) are validated strictly (present, integer,
  and equal to the source rows); a missing/wrong-type/wrong-value count fails;
* a **trusted expected source sha256 is REQUIRED** (``--expected-source-sha256`` /
  an independent checklist file / ``expected_source_sha256``). The expected value
  is compared to the file's real hash — we never compute a hash from the file and
  then claim it as expected. Missing binding -> ``failed`` /
  ``reason="source_binding_required"`` (no data written); a mismatch is rejected
  (``source_mismatch``). There is NO insecure default: a manifest-only binding is
  not enough, and a legacy record already imported before this rule is kept as-is;
* a successful import records ``source_binding = "verified"`` (the DB expected hash
  matched). ``history_status`` still reports legacy records imported before this
  rule as ``source_binding="unbound"`` and ``state="ok"`` (existing read-only
  display preserved; no re-import or revoke required);
* import is **atomic**: the retrieval rows AND the success metadata are written in
  ONE transaction and committed together. Any ID/field conflict, SQL error, or
  application error rolls the whole thing back; a failure record is written in a
  SEPARATE transaction. There is never success metadata without the data (or
  data without success metadata) presented as ``ok``;
* a row whose primary key matches is compared on ALL 11 evidence fields; any
  difference (message_id, parent_id, content_type, has_attachment, role, text,
  create_time, ...) is a conflict that rolls the whole batch back;
* the same source hash is **idempotent** (no duplicates);
* imported history goes ONLY into ``messages``/``conversations`` — it is never
  written into ``chat_log`` and never auto-enqueued into the maintenance queue;
* independent metadata records source hashes, counts, and the time range, with an
  explicit ``summary_covered_range = "unknown"`` (we never claim full coverage);
* ``summary_covered_range`` stays ``"unknown"`` unless a constrained local
  coverage marker (see ``coverage.py``) is present AND verifiable against this
  user's published summary and the stored successful-import hashes. The marker is
  never created here; only the main Agent's bootstrap writes it, and even then it
  only names the processed range, never "all details are remembered".

CLI:
    python3 import_history.py --source-db <retrieval_import.sqlite> \\
                              --manifest <manifest.json> \\
                              [--expected-source-sha256 <hex>] [--expected-hashes <json>] \\
                              [--db <trial sqlite>] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import db
from .coverage import compute_coverage, latest_ok_import
from .config import Config, load_config
from .retrieval import now_shanghai
from .summary_store import read_published
from .util import sha256_file

EXPECTED_TOOL = "history-bootstrap-v1"
IMPORTABLE_ROLES = ("user", "assistant")
REQUIRED_MESSAGE_COLUMNS = [
    "conversation_id", "node_id", "message_id", "parent_id", "role", "content_type",
    "text", "create_time", "is_selected_branch", "is_assistant_reasoning", "has_attachment",
]
REQUIRED_CONVERSATION_COLUMNS = [
    "id", "title", "create_time", "update_time", "current_node", "node_count",
    "message_count", "source_file",
]
# manifest counts that MUST be present, integer-typed, and match the source exactly
REQUIRED_COUNTS = ("conversations", "evidence_messages", "user_messages", "assistant_messages")
SUMMARY_COVERAGE_UNKNOWN = "unknown"
_IMPORT_META_COLUMNS = ("source_db_sha256_expected", "manifest_sha256_expected", "source_binding")


class HistoryImportNotConfigured(RuntimeError):
    """No retrieval source was configured for the import."""


class HistoryImportConflict(RuntimeError):
    """A source row collides with existing content (id/field conflict, refused)."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_only_connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())


# --------------------------------------------------------------------------- #
# Trusted source binding (independent of the file being scanned)
# --------------------------------------------------------------------------- #
def load_expected_hashes(checklist: Path | str | None) -> dict[str, str]:
    """Read an independent expected-hash checklist.

    Accepted shape: ``{"source_db_sha256": "<hex>", "manifest_sha256": "<hex>"}``
    (``source_sha256`` is accepted as an alias for the source). The values are
    TRUSTED expectations authored outside this importer — never computed from the
    file under test.
    """
    if not checklist:
        return {}
    data = json.loads(Path(checklist).read_text("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("expected-hashes checklist must be a JSON object")
    out: dict[str, str] = {}
    src = data.get("source_db_sha256") or data.get("source_sha256")
    man = data.get("manifest_sha256")
    if src:
        out["source_db_sha256"] = str(src)
    if man:
        out["manifest_sha256"] = str(man)
    return out


def resolve_expected_hashes(*, expected_source_sha256: str | None = None,
                            expected_manifest_sha256: str | None = None,
                            checklist: Path | str | None = None) -> tuple[str | None, str | None]:
    """Resolve trusted expected hashes (explicit args win over an optional checklist)."""
    from_checklist = load_expected_hashes(checklist)
    src = expected_source_sha256 or from_checklist.get("source_db_sha256")
    man = expected_manifest_sha256 or from_checklist.get("manifest_sha256")
    return src, man


# --------------------------------------------------------------------------- #
# Manifest validation
# --------------------------------------------------------------------------- #
def _counts_error(counts: Any) -> str | None:
    """Strict presence + integer-type check for the REQUIRED counts."""
    if not isinstance(counts, dict):
        return "counts 不是对象"
    for key in REQUIRED_COUNTS:
        if key not in counts:
            return f"counts 缺少 {key}"
        value = counts[key]
        if isinstance(value, bool) or not isinstance(value, int):
            return f"counts.{key} 不是整数（{type(value).__name__}）"
    return None


def _counts_mismatch(counts: dict, *, conversations: int, messages: int, users: int,
                     assistants: int) -> str | None:
    if counts["conversations"] != conversations:
        return f"counts.conversations={counts['conversations']} != 源 conversations 行数 {conversations}"
    if counts["evidence_messages"] != messages:
        return f"counts.evidence_messages={counts['evidence_messages']} != 源 messages 行数 {messages}"
    if counts["user_messages"] != users:
        return f"counts.user_messages={counts['user_messages']} != 源 user 行数 {users}"
    if counts["assistant_messages"] != assistants:
        return f"counts.assistant_messages={counts['assistant_messages']} != 源 assistant 行数 {assistants}"
    return None


# --------------------------------------------------------------------------- #
# Metadata helpers (no implicit commits; callers own the transaction)
# --------------------------------------------------------------------------- #
_REC_KEYS = ("source_id", "status", "source_db", "source_db_sha256", "manifest",
             "manifest_sha256", "imported_at", "messages", "conversations", "users",
             "assistants", "skipped", "undated", "time_min", "time_max",
             "summary_covered_range", "error", "note",
             "source_db_sha256_expected", "manifest_sha256_expected", "source_binding")


def _ensure_meta_columns(conn: sqlite3.Connection) -> None:
    cols = _columns(conn, "history_imports")
    added = False
    for name in _IMPORT_META_COLUMNS:
        if name not in cols:
            conn.execute(f"ALTER TABLE history_imports ADD COLUMN {name} TEXT")
            added = True
    if added:
        conn.commit()


def _insert_import_row(conn: sqlite3.Connection, fields: dict[str, Any]) -> None:
    """Insert/replace one ``history_imports`` row WITHOUT committing."""
    fields = dict(fields)
    fields.setdefault("status", "ok")
    fields.setdefault("summary_covered_range", SUMMARY_COVERAGE_UNKNOWN)
    fields.setdefault("imported_at", _utc_now())
    values = [fields.get(k) for k in _REC_KEYS]
    conn.execute(
        f"INSERT OR REPLACE INTO history_imports({','.join(_REC_KEYS)})"
        f" VALUES({','.join('?' * len(_REC_KEYS))})",
        values,
    )


def _record(conn: sqlite3.Connection, **fields: Any) -> None:
    """Insert/replace one metadata row (same transaction as the caller)."""
    _insert_import_row(conn, fields)


def _write_failure(conn: sqlite3.Connection, **fields: Any) -> None:
    """Write a failure record in its OWN transaction after a rollback.

    Uses ``_insert_import_row`` directly (not ``_record``) so a fault injected
    into ``_record`` still leaves an auditable failure row; if even that fails the
    error is swallowed (the important invariant — no false success — already holds).
    """
    try:
        conn.rollback()
    except sqlite3.Error:
        pass
    try:
        _insert_import_row(conn, {**fields, "status": "failed"})
        conn.commit()
    except sqlite3.Error:
        try:
            conn.rollback()
        except sqlite3.Error:
            pass


def _fail(conn: sqlite3.Connection | None, *, source_db, manifest, source_db_sha, manifest_sha,
          source_id, error, note: str = "", counts: dict | None = None, reason: str = "failed",
          expected_source_sha: str | None = None, manifest_sha_expected: str | None = None,
          source_binding: str = "unbound") -> dict:
    counts = counts or {}
    if conn is not None and source_id:
        _write_failure(
            conn, source_id=source_id, source_db=str(source_db) if source_db else None,
            source_db_sha256=source_db_sha, manifest=str(manifest) if manifest else None,
            manifest_sha256=manifest_sha, messages=counts.get("evidence_messages"),
            conversations=counts.get("conversations"), error=error, note=note,
            source_db_sha256_expected=expected_source_sha,
            manifest_sha256_expected=manifest_sha_expected, source_binding=source_binding)
    return {"status": "failed", "state": "failed", "reason": reason, "error": error, "note": note,
            "source_id": source_id, "messages": 0, "conversations": 0,
            "source_binding": source_binding,
            "summary_covered_range": SUMMARY_COVERAGE_UNKNOWN}


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def import_history(config: Config, *, source_db: Path | str | None = None,
                   manifest: Path | str | None = None,
                   conn: sqlite3.Connection | None = None, force: bool = False,
                   expected_source_sha256: str | None = None,
                   expected_manifest_sha256: str | None = None,
                   expected_hashes: Path | str | None = None) -> dict[str, Any]:
    """Import a ``retrieval_import.sqlite`` into this trial's retrieval tables.

    Returns a dict with ``status`` in {ok, noop, failed}. Expected failures
    (missing/bad source, bad manifest, unbound source, ID/field conflict, any
    SQL/application error) are reported, never raised.
    """
    src_raw = source_db if source_db is not None else config.retrieval_source_db
    man_raw = manifest if manifest is not None else config.retrieval_manifest
    src_path = Path(src_raw) if src_raw else None
    man_path = Path(man_raw) if man_raw else None
    own = conn is None
    if own:
        conn = db.connect(config.db_path)
        db.init_db(conn)
    try:
        _ensure_meta_columns(conn)
        return _import(config, conn, src_path, man_path, force=force,
                       expected_source_sha256=expected_source_sha256,
                       expected_manifest_sha256=expected_manifest_sha256,
                       expected_hashes=expected_hashes)
    finally:
        if own:
            conn.close()


def _import(config: Config, conn: sqlite3.Connection, src_path: Path | None, man_path: Path | None,
            *, force: bool, expected_source_sha256: str | None = None,
            expected_manifest_sha256: str | None = None,
            expected_hashes: Path | str | None = None) -> dict[str, Any]:
    if src_path is None or not src_path.exists():
        return _fail(conn, source_db=src_path, manifest=man_path, source_db_sha=None,
                     manifest_sha=None, source_id=None, reason="not_configured",
                     error="未配置或找不到 retrieval_import.sqlite（--source-db）",
                     note="本试用只读独立检索源；可稍后由主 Agent 导入。")
    if man_path is None or not man_path.exists():
        return _fail(conn, source_db=src_path, manifest=man_path, source_db_sha=None,
                     manifest_sha=None, source_id=None,
                     error="未配置或找不到 manifest.json（--manifest）",
                     note="导入需要来源 manifest 才能核对。")
    # hashes/source_id computed early so EVERY failure can write an auditable record
    db_sha = sha256_file(src_path)
    man_sha = sha256_file(man_path)
    source_id = hashlib.sha256(f"db:{db_sha}|manifest:{man_sha}".encode()).hexdigest()[:16]
    try:
        manifest = json.loads(man_path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _fail(conn, source_db=src_path, manifest=man_path, source_db_sha=db_sha,
                     manifest_sha=man_sha, source_id=source_id, error=f"manifest 不是合法 JSON：{exc}")
    if not isinstance(manifest, dict) or manifest.get("tool") != EXPECTED_TOOL:
        return _fail(conn, source_db=src_path, manifest=man_path, source_db_sha=db_sha,
                     manifest_sha=man_sha, source_id=source_id,
                     error=f"manifest.tool 不是 {EXPECTED_TOOL!r}（错源）")
    if manifest.get("status") != "ok":
        return _fail(conn, source_db=src_path, manifest=man_path, source_db_sha=db_sha,
                     manifest_sha=man_sha, source_id=source_id,
                     error=f"manifest.status={manifest.get('status')!r}（源构建未成功）")

    # ---- strict count presence + integer type (missing / bad type -> failed) ----
    counts = manifest.get("counts")
    cerr = _counts_error(counts)
    if cerr:
        return _fail(conn, source_db=src_path, manifest=man_path, source_db_sha=db_sha,
                     manifest_sha=man_sha, source_id=source_id, reason="bad_manifest",
                     error=f"坏 manifest：{cerr}", note="必要计数缺失或类型不对，不作导入。")
    counts = dict(counts)

    # ---- resolve the trusted expected hashes (source binding enforced after counts) ----
    try:
        exp_src, exp_man = resolve_expected_hashes(
            expected_source_sha256=expected_source_sha256,
            expected_manifest_sha256=expected_manifest_sha256, checklist=expected_hashes)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _fail(conn, source_db=src_path, manifest=man_path, source_db_sha=db_sha,
                     manifest_sha=man_sha, source_id=source_id, counts=counts,
                     reason="bad_checklist", source_binding="required",
                     error=f"expected-hashes 清单不可用：{exc}")
    if exp_src is None:
        binding = "required"
    elif not _is_sha256(exp_src) or db_sha != exp_src.lower():
        binding = "mismatch" if _is_sha256(exp_src) else "required"
    else:
        binding = "verified"

    # ---- open the source read-only and validate its shape ----
    try:
        src = _read_only_connect(src_path)
    except sqlite3.Error as exc:
        return _fail(conn, source_db=src_path, manifest=man_path, source_db_sha=db_sha,
                     manifest_sha=man_sha, source_id=source_id, counts=counts,
                     expected_source_sha=exp_src, manifest_sha_expected=exp_man,
                     source_binding=binding, error=f"无法只读打开源 sqlite：{exc}")
    try:
        tables = _table_names(src)
        if not {"messages", "conversations"} <= tables:
            return _fail(conn, source_db=src_path, manifest=man_path, source_db_sha=db_sha,
                         manifest_sha=man_sha, source_id=source_id, counts=counts,
                         expected_source_sha=exp_src, manifest_sha_expected=exp_man,
                         source_binding=binding, error="源 sqlite 缺少 messages/conversations 表（错源）")
        if not set(REQUIRED_MESSAGE_COLUMNS) <= _columns(src, "messages"):
            return _fail(conn, source_db=src_path, manifest=man_path, source_db_sha=db_sha,
                         manifest_sha=man_sha, source_id=source_id, counts=counts,
                         expected_source_sha=exp_src, manifest_sha_expected=exp_man,
                         source_binding=binding, error="源 messages 表列不匹配（错源）")
        if not set(REQUIRED_CONVERSATION_COLUMNS) <= _columns(src, "conversations"):
            return _fail(conn, source_db=src_path, manifest=man_path, source_db_sha=db_sha,
                         manifest_sha=man_sha, source_id=source_id, counts=counts,
                         expected_source_sha=exp_src, manifest_sha_expected=exp_man,
                         source_binding=binding, error="源 conversations 表列不匹配（错源）")
        src_msgs = [dict(zip(REQUIRED_MESSAGE_COLUMNS, row)) for row in src.execute(
            f"SELECT {','.join(REQUIRED_MESSAGE_COLUMNS)} FROM messages")]
        src_convs = [dict(zip(REQUIRED_CONVERSATION_COLUMNS, row)) for row in src.execute(
            f"SELECT {','.join(REQUIRED_CONVERSATION_COLUMNS)} FROM conversations")]
    except sqlite3.Error as exc:
        return _fail(conn, source_db=src_path, manifest=man_path, source_db_sha=db_sha,
                     manifest_sha=man_sha, source_id=source_id, counts=counts,
                     expected_source_sha=exp_src, manifest_sha_expected=exp_man,
                     source_binding=binding, error=f"读取源 sqlite 失败：{exc}")
    finally:
        src.close()

    # ---- cross-check the manifest counts against the source rows ----
    n_users = sum(1 for m in src_msgs if m["role"] == "user")
    n_assist = sum(1 for m in src_msgs if m["role"] == "assistant")
    mismatch = _counts_mismatch(counts, conversations=len(src_convs), messages=len(src_msgs),
                                users=n_users, assistants=n_assist)
    if mismatch:
        return _fail(conn, source_db=src_path, manifest=man_path, source_db_sha=db_sha,
                     manifest_sha=man_sha, source_id=source_id, counts=counts,
                     expected_source_sha=exp_src, manifest_sha_expected=exp_man,
                     source_binding=binding, reason="bad_manifest",
                     error=f"坏 manifest：{mismatch}",
                     note="manifest 计数与源不一致，拒绝导入。")

    # ---- v4.2: a trusted expected SOURCE hash is REQUIRED (no insecure default) ----
    # Checked after the manifest's own self-consistency: a valid source with no
    # binding is refused (source_binding_required) and writes NO data; a same-source
    # retry also requires the binding (no unbound noop); a manifest-only binding is
    # not enough (the DB identity must be bound).
    if exp_src is None:
        return _fail(conn, source_db=src_path, manifest=man_path, source_db_sha=db_sha,
                     manifest_sha=man_sha, source_id=source_id, counts=counts,
                     reason="source_binding_required", source_binding="required",
                     error=("必须提供受信预期来源 DB sha256（--expected-source-sha256 或独立 "
                            "--expected-hashes 清单）；不接受无绑定的默认导入（仅绑定 manifest 不足）。"),
                     note="拒绝无来源绑定的导入；检索数据与成功 metadata 均未写入。")
    if not _is_sha256(exp_src):
        return _fail(conn, source_db=src_path, manifest=man_path, source_db_sha=db_sha,
                     manifest_sha=man_sha, source_id=source_id, counts=counts,
                     reason="bad_expected", expected_source_sha=exp_src, source_binding="required",
                     error="受信预期来源 hash 不是 64 位十六进制")
    if db_sha != exp_src.lower():
        return _fail(conn, source_db=src_path, manifest=man_path, source_db_sha=db_sha,
                     manifest_sha=man_sha, source_id=source_id, counts=counts,
                     reason="source_mismatch", expected_source_sha=exp_src.lower(),
                     source_binding="mismatch",
                     error="来源 sqlite hash 与受信预期不符（同数量错配来源被拒绝）")
    if exp_man is not None:
        if not _is_sha256(exp_man):
            return _fail(conn, source_db=src_path, manifest=man_path, source_db_sha=db_sha,
                         manifest_sha=man_sha, source_id=source_id, counts=counts,
                         reason="bad_expected", expected_source_sha=exp_src,
                         manifest_sha_expected=exp_man, source_binding="verified",
                         error="受信预期 manifest hash 不是 64 位十六进制")
        if man_sha != exp_man.lower():
            return _fail(conn, source_db=src_path, manifest=man_path, source_db_sha=db_sha,
                         manifest_sha=man_sha, source_id=source_id, counts=counts,
                         reason="manifest_mismatch", expected_source_sha=exp_src,
                         manifest_sha_expected=exp_man.lower(), source_binding="verified",
                         error="manifest hash 与受信预期不符（错配 manifest 被拒绝）")

    # ---- idempotency (checked AFTER the binding gate: a noop also needs the binding) ----
    existing_ok = conn.execute(
        "SELECT source_id FROM history_imports WHERE source_id=? AND status='ok'", (source_id,)
    ).fetchone()
    if existing_ok and not force:
        live = _live_stats(conn)
        return {"status": "noop", "state": "ok", "reason": "same source already imported",
                "source_id": source_id, "duplicate": True, "source_binding": binding, **live,
                "summary_covered_range": SUMMARY_COVERAGE_UNKNOWN}

    # ---- defensive guards: only selected-branch, non-reasoning, user/assistant, non-empty ----
    kept: list[dict] = []
    skipped = {"reasoning": 0, "not_selected": 0, "other_role": 0, "empty": 0}
    for r in src_msgs:
        if int(r["is_assistant_reasoning"] or 0) == 1:
            skipped["reasoning"] += 1
            continue
        if int(r["is_selected_branch"] or 0) != 1:
            skipped["not_selected"] += 1
            continue
        if (r["role"] or "") not in IMPORTABLE_ROLES:
            skipped["other_role"] += 1
            continue
        if not (r["text"] or "").strip():
            skipped["empty"] += 1
            continue
        kept.append(r)
    total_skipped = sum(skipped.values())

    # ---- atomic import: data + success metadata in ONE transaction ----
    live: dict[str, Any] | None = None
    try:
        if not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE")
        _apply(conn, src_convs, kept)                 # no commit inside
        live = _live_stats(conn)
        _record(conn, source_id=source_id, status="ok", source_db=str(src_path),
                source_db_sha256=db_sha, manifest=str(man_path), manifest_sha256=man_sha,
                messages=live["messages"], conversations=live["conversations"],
                users=live["users"], assistants=live["assistants"], skipped=total_skipped,
                undated=live["undated"], time_min=live["time_min_epoch"], time_max=live["time_max_epoch"],
                summary_covered_range=SUMMARY_COVERAGE_UNKNOWN,
                source_db_sha256_expected=exp_src, manifest_sha256_expected=exp_man,
                source_binding=binding,
                note=("历史导入仅用于检索；未写入 chat_log，未进入维护队列；"
                      f"summary 覆盖范围未知（不推断全覆盖）；跳过 {skipped}；"
                      f"来源绑定 {binding}"))
        conn.commit()
    except HistoryImportConflict as exc:
        _write_failure(conn, source_id=source_id, source_db=str(src_path), source_db_sha256=db_sha,
                       manifest=str(man_path), manifest_sha256=man_sha,
                       messages=counts.get("evidence_messages"),
                       conversations=counts.get("conversations"), error=str(exc),
                       source_db_sha256_expected=exp_src, manifest_sha256_expected=exp_man,
                       source_binding=binding,
                       note=(f"拒绝导入：{exc}；本批全部回滚，未写入任何行，也未改写 "
                             "chat_log/队列/摘要。"))
        return {"status": "failed", "state": "failed", "reason": "conflict", "source_id": source_id,
                "error": str(exc), "messages": 0, "conversations": 0, "source_binding": binding,
                "summary_covered_range": SUMMARY_COVERAGE_UNKNOWN}
    except Exception as exc:  # noqa: BLE001  any SQL/application error -> rollback, no false success
        _write_failure(conn, source_id=source_id, source_db=str(src_path), source_db_sha256=db_sha,
                       manifest=str(man_path), manifest_sha256=man_sha,
                       messages=counts.get("evidence_messages"),
                       conversations=counts.get("conversations"),
                       error=f"{type(exc).__name__}: {exc}",
                       source_db_sha256_expected=exp_src, manifest_sha256_expected=exp_man,
                       source_binding=binding,
                       note="导入过程出现 SQL/应用异常；本批全部回滚（数据与成功元数据均未提交）。")
        return {"status": "failed", "state": "failed", "reason": "error", "source_id": source_id,
                "error": f"{type(exc).__name__}: {exc}", "messages": 0, "conversations": 0,
                "source_binding": binding, "summary_covered_range": SUMMARY_COVERAGE_UNKNOWN}

    return {"status": "ok", "state": "ok", "source_id": source_id, "duplicate": False,
            "skipped": skipped, "source_binding": binding,
            "summary_covered_range": SUMMARY_COVERAGE_UNKNOWN, **(live or {})}


def _msg_sig(d: dict) -> tuple:
    return (
        d["conversation_id"], d["node_id"], d["message_id"], d["parent_id"], d["role"],
        d["content_type"], d["text"], d["create_time"],
        1 if int(d["is_selected_branch"] or 0) else 0,
        1 if int(d["is_assistant_reasoning"] or 0) else 0,
        1 if d["has_attachment"] else 0,
    )


def _apply(conn: sqlite3.Connection, src_convs: list[dict], kept: list[dict]) -> None:
    """Insert imported rows inside ONE transaction; raise on any ID/field conflict.

    A row whose primary key (conversation_id, node_id) already exists is compared
    on ALL 11 evidence fields; any difference is a conflict. Never commits here.
    """
    cols = ",".join(REQUIRED_MESSAGE_COLUMNS)
    existing_msg = {
        (r["conversation_id"], r["node_id"]): _msg_sig(dict(r))
        for r in conn.execute(f"SELECT {cols} FROM messages")
    }
    existing_mid: dict[str, tuple[str, str]] = {}
    for r in conn.execute("SELECT conversation_id,node_id,message_id FROM messages"):
        if r["message_id"]:
            existing_mid.setdefault(r["message_id"], (r["conversation_id"], r["node_id"]))
    existing_conv = {
        r["id"]: (r["title"], r["create_time"], r["update_time"], r["current_node"],
                  r["node_count"], r["message_count"], r["source_file"])
        for r in conn.execute("SELECT id,title,create_time,update_time,current_node,node_count,"
                              "message_count,source_file FROM conversations")
    }
    for c in src_convs:
        sig = (c["title"], c["create_time"], c["update_time"], c["current_node"],
               c["node_count"], c["message_count"], c["source_file"])
        if c["id"] in existing_conv:
            if existing_conv[c["id"]] != sig:
                raise HistoryImportConflict(f"conversation id 冲突：{c['id']} 已存在且内容不同")
            continue
        conn.execute("INSERT INTO conversations(id,title,create_time,update_time,current_node,"
                     "node_count,message_count,source_file) VALUES(?,?,?,?,?,?,?,?)",
                     (c["id"], c["title"], c["create_time"], c["update_time"], c["current_node"],
                      c["node_count"], c["message_count"], c["source_file"]))
        existing_conv[c["id"]] = sig
    for r in kept:
        key = (r["conversation_id"], r["node_id"])
        sig = _msg_sig(r)
        if key in existing_msg:
            if existing_msg[key] != sig:
                raise HistoryImportConflict(
                    f"message 字段冲突：{r['conversation_id']}#{r['node_id']} 已存在且 11 字段不一致")
            continue
        mid = r["message_id"]
        if mid and mid in existing_mid and existing_mid[mid] != key:
            raise HistoryImportConflict(
                f"message_id 冲突：{mid} 已属于 {existing_mid[mid]}，拒绝静默覆盖")
        conn.execute(
            "INSERT INTO messages(conversation_id,node_id,message_id,parent_id,role,content_type,"
            "text,create_time,is_selected_branch,is_assistant_reasoning,has_attachment)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (r["conversation_id"], r["node_id"], r["message_id"], r["parent_id"], r["role"],
             r["content_type"], r["text"], r["create_time"], 1, 0, 1 if r["has_attachment"] else 0))
        existing_msg[key] = sig
        if mid:
            existing_mid.setdefault(mid, key)


def _live_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    msgs = conn.execute("SELECT COUNT(*) c FROM messages").fetchone()["c"]
    convs = conn.execute("SELECT COUNT(*) c FROM conversations").fetchone()["c"]
    users = conn.execute("SELECT COUNT(*) c FROM messages WHERE role='user'").fetchone()["c"]
    assistants = conn.execute("SELECT COUNT(*) c FROM messages WHERE role='assistant'").fetchone()["c"]
    undated = conn.execute("SELECT COUNT(*) c FROM messages WHERE create_time IS NULL").fetchone()["c"]
    rng = conn.execute("SELECT MIN(create_time) lo, MAX(create_time) hi FROM messages "
                       "WHERE create_time IS NOT NULL").fetchone()
    lo, hi = rng["lo"], rng["hi"]
    return {
        "messages": msgs, "conversations": convs, "users": users, "assistants": assistants,
        "undated": undated,
        "time_min_epoch": lo, "time_max_epoch": hi,
        "time_min": now_shanghai(lo) if lo is not None else None,
        "time_max": now_shanghai(hi) if hi is not None else None,
    }


def history_status(config: Config, conn: sqlite3.Connection | None = None) -> dict[str, Any]:
    """Retrieval-history state for /health and the UI (never claims full coverage)."""
    own = conn is None
    if own:
        conn = db.connect(config.db_path)
        db.init_db(conn)
    try:
        latest = conn.execute(
            "SELECT * FROM history_imports ORDER BY rowid DESC LIMIT 1").fetchone()
        live = _live_stats(conn)
        # Constrained coverage claim: evaluated from a local marker (never written
        # here) against this user's published summary AND the stored successful-import
        # hashes. Unknown unless every gate passes; only names the processed range.
        coverage = compute_coverage(
            config,
            live={"messages": live["messages"], "conversations": live["conversations"],
                  "time_min": live["time_min_epoch"], "time_max": live["time_max_epoch"]},
            latest=latest_ok_import(conn),
            published=read_published(config.maint_state_dir, config.user_id),
        )
        covered_range = coverage["range_text"] if coverage["state"] == "covered" else None
        imported = live["messages"] > 0
        base = {
            "implemented": True,
            "source_db": str(config.retrieval_source_db) if config.retrieval_source_db else None,
            "manifest": str(config.retrieval_manifest) if config.retrieval_manifest else None,
            "summary_covered_range": covered_range or SUMMARY_COVERAGE_UNKNOWN,
            "summary_coverage": coverage,
            "summary_note": ("常驻摘要只声明标记过的已处理范围，不保证记住全部细节；"
                             "历史导入不会自动进入维护队列。"),
            **live,
        }
        if latest is None and not imported:
            return {**base, "state": "none", "imported": False, "source_binding": "none",
                    "note": "尚未导入检索历史；可由主 Agent 用 import_history.py 导入。"}
        if latest is None and imported:
            # data present but NO success metadata -> must NOT be shown as ok
            return {**base, "state": "inconsistent", "imported": True, "source_binding": "none",
                    "error": "检索表有数据，但没有任何成功导入元数据（不视为成功）",
                    "note": "数据与成功元数据不一致；请重新导入。"}
        if latest["status"] == "failed":
            return {**base, "state": "failed", "imported": imported,
                    "source_id": latest["source_id"], "source_db": latest["source_db"],
                    "manifest": latest["manifest"], "error": latest["error"],
                    "imported_at": latest["imported_at"],
                    "source_binding": _col(latest, "source_binding") or "unbound",
                    "note": latest["note"] or "上次导入失败。"}
        if not imported:
            return {**base, "state": "failed", "imported": False, "source_id": latest["source_id"],
                    "error": "元数据标记为成功，但检索表中没有记录（数据不一致）",
                    "note": "请重新导入。"}
        return {**base, "state": "ok", "imported": True,
                "source_id": latest["source_id"] if latest else None,
                "source_db": latest["source_db"] if latest else base["source_db"],
                "manifest": latest["manifest"] if latest else base["manifest"],
                "source_db_sha256": latest["source_db_sha256"] if latest else None,
                "manifest_sha256": latest["manifest_sha256"] if latest else None,
                "source_binding": _col(latest, "source_binding") or "unbound",
                "imported_at": latest["imported_at"] if latest else None,
                "note": (f"历史检索可用（来源绑定 {_col(latest, 'source_binding') or 'unbound'}）；"
                         "仅用于检索，未写入会话，未进入维护队列。")}
    finally:
        if own:
            conn.close()


def _col(row: sqlite3.Row, name: str) -> Any:
    try:
        return row[name]
    except (IndexError, KeyError):
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="import_history.py",
                                 description="Import a read-only retrieval_import.sqlite (0 API)")
    ap.add_argument("--source-db", required=False, help="retrieval_import.sqlite (read-only)")
    ap.add_argument("--manifest", required=False, help="manifest.json from history-bootstrap-v1")
    ap.add_argument("--db", default=None, help="override the trial chat sqlite path")
    ap.add_argument("--expected-source-sha256", default=None,
                    help="REQUIRED (or --expected-hashes): trusted expected sha256 of the source DB, "
                         "independent of the file; a missing binding is refused")
    ap.add_argument("--expected-manifest-sha256", default=None,
                    help="trusted expected sha256 of the manifest")
    ap.add_argument("--expected-hashes", default=None,
                    help="path to an independent expected-hash checklist JSON "
                         "({\"source_db_sha256\":..,\"manifest_sha256\":..}); a source hash is REQUIRED")
    ap.add_argument("--force", action="store_true", help="re-verify even if the same source hash was imported")
    args = ap.parse_args(argv)

    config = load_config()
    if args.db:
        config.db_path = Path(args.db)
    config.db_path.parent.mkdir(parents=True, exist_ok=True)  # only the db is needed here
    conn = db.connect(config.db_path)
    db.init_db(conn)
    try:
        result = import_history(config, source_db=args.source_db, manifest=args.manifest,
                                conn=conn, force=args.force,
                                expected_source_sha256=args.expected_source_sha256,
                                expected_manifest_sha256=args.expected_manifest_sha256,
                                expected_hashes=args.expected_hashes)
    finally:
        conn.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in ("ok", "noop") else 2


if __name__ == "__main__":
    raise SystemExit(main())
