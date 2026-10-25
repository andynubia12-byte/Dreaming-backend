"""Constrained LOCAL coverage marker for the trial's history bootstrap.

The 10230-row history import only makes the raw turns *retrievable*; standing
understanding is produced later by a separate bootstrap that walks the imported
source in groups and publishes memory versions. This module is the only place that
can turn "the bootstrap finished (part of) the history" into a displayable claim,
and the claim is deliberately narrow:

* the marker is a plain JSON file **inside the trial's ``runs/`` directory** — it is
  never created automatically, never written by chat, the importer, retrieval or
  the maintenance queue, and it never enqueues work;
* ``history_status`` shows "常驻摘要已处理的历史范围" ONLY when ALL of these hold:
  the marker parses with the expected schema, its ``user_id`` equals this trial's
  user, its source hashes match the **stored** hashes of the current successful
  import metadata (when stored has a manifest hash, the marker MUST carry an equal
  one — an absent field is not a bypass), the counts/time range are structurally
  valid (positive counts, within the live import, not reversed, not in the future),
  AND a real published summary exists for that same user with
  ``version >= memory_version``;
* anything missing or untrustworthy (no marker / corrupt marker / wrong user /
  wrong source / no published summary / a marker ahead of the published version /
  a missing or unverifiable recorded summary hash / reversed or future range / bad
  counts) stays ``unknown`` with a diagnosable ``reason``; the UI keeps showing
  "未知";
* when the published version has advanced past the marker, the recorded summary
  hash must still match an **immutable per-version release** of that same user (the
  engine never overwrites a published version directory), so a missing or forged
  old hash is not accepted merely because the current version number is larger; a
  genuine later publication keeps the legitimate older marker valid;
* a ``complete=true`` marker must cover the FULL dated range of the current import
  (within a small float tolerance), not a narrow interior slice;
* even a valid marker only claims the **processed range**, never "all details are
  remembered" — it is a coverage pointer, not a completeness guarantee.

``write_coverage_marker`` is the atomic writer helper the main Agent's bootstrap
calls after a successful publish. It refuses inputs it cannot verify against the
database and the published state: it will not accept a bare version string and
declare the history processed. A rejected or failed write leaves the previous
marker untouched (or no marker at all).
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import db
from .config import Config
from .retrieval import now_shanghai
from .summary_store import Published, read_published

SCHEMA = "continuity-coverage-marker-v1"
MARKER_FILENAME = "history_coverage_marker.json"

SUMMARY_COVERAGE_UNKNOWN = "unknown"
COVERED = "covered"
PARTIAL = "partial"
UNKNOWN = "unknown"

# Absolute tolerance for epoch comparisons. Source epochs are copied verbatim from
# the import (the same float64), so a truthful claim matches to the last bit; this
# tolerance is float-write precision only (~2.4e-7 s at 1.6e9 epochs), far below the
# 1-second scale where a claim would actually be wrong. It never absorbs a narrower
# or out-of-bounds range.
RANGE_TOLERANCE_SECONDS = 1e-6

# Human-readable, diagnosable reasons (kept stable; the UI shows the raw state).
REASON_TEXT = {
    "marker_missing": "尚无初始化标记（常驻理解未初始化）",
    "marker_unreadable": "初始化标记不可读（非合法 JSON 或缺字段）",
    "marker_schema_mismatch": "初始化标记 schema 不匹配",
    "marker_user_mismatch": "初始化标记属于其它用户",
    "no_successful_import": "没有成功的检索历史导入元数据",
    "source_mismatch": "初始化标记的来源 hash 与当前成功导入不符",
    "source_unbound": "成功导入元数据缺少来源 hash，无法绑定",
    "no_published_summary": "本用户尚无已发布摘要",
    "summary_version_stale": "初始化标记版本高于已发布摘要（过期/超前标记）",
    "summary_hash_mismatch": "初始化标记记录的摘要 hash 与已发布/历史摘要不符",
    "summary_hash_missing": "初始化标记缺少摘要 hash，无法核对",
    "history_version_missing": "标记版本对应该用户的历史发布记录缺失（无法核对）",
    "counts_invalid": "初始化标记的已处理数量无效或超出当前导入",
    "range_invalid": "初始化标记的时间范围无效（倒序/越界/未来/完整声明未覆盖全量）",
    "coverage_error": "coverage 计算异常",
}


def marker_path(config: Config, explicit: Path | str | None = None) -> Path:
    """The single marker location: inside the trial's ``runs/`` directory."""
    if explicit is not None:
        return Path(explicit)
    return Path(config.runs_dir) / MARKER_FILENAME


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
        return None
    return float(value)


def _date_text(epoch: Any) -> str | None:
    value = _num(epoch)
    if value is None:
        return None
    try:
        return now_shanghai(value)[:10]
    except (OSError, OverflowError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Atomic write (temp file in the same dir -> fsync -> os.replace)
# --------------------------------------------------------------------------- #
def _atomic_write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    try:  # best-effort directory durability; a failure here does not invalidate the write
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# Shared evidence readers (the writer owns its own connection)
# --------------------------------------------------------------------------- #
def _live_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    msgs = conn.execute("SELECT COUNT(*) c FROM messages").fetchone()["c"]
    convs = conn.execute("SELECT COUNT(*) c FROM conversations").fetchone()["c"]
    rng = conn.execute("SELECT MIN(create_time) lo, MAX(create_time) hi FROM messages "
                       "WHERE create_time IS NOT NULL").fetchone()
    return {"messages": msgs, "conversations": convs, "time_min": rng["lo"], "time_max": rng["hi"]}


def latest_ok_import(conn: sqlite3.Connection) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM history_imports WHERE status='ok' "
                       "ORDER BY rowid DESC LIMIT 1").fetchone()
    return dict(row) if row is not None else None


# --------------------------------------------------------------------------- #
# Claim validation (shared by the reader and the writer)
# --------------------------------------------------------------------------- #
def _validate_claims(*, processed: Any, time_min: Any, time_max: Any, complete: Any,
                     live: dict[str, Any], now: float) -> tuple[bool, str, str, dict[str, Any]]:
    """Validate structure/time/counts against the live import. Never raises."""
    if not isinstance(complete, bool):
        return False, "marker_unreadable", "complete 不是布尔值", {}
    if not isinstance(processed, dict):
        return False, "counts_invalid", "processed 不是对象", {}
    msgs = processed.get("messages")
    convs = processed.get("conversations")
    if not _is_int(msgs) or not _is_int(convs) or msgs <= 0 or convs <= 0:
        return False, "counts_invalid", "processed.messages/conversations 不是正整数", {}
    if msgs > live["messages"] or convs > live["conversations"]:
        return False, "counts_invalid", "已处理数量超过当前导入", {}
    if complete and (msgs != live["messages"] or convs != live["conversations"]):
        return False, "counts_invalid", "标记 complete 但已处理数量与当前导入不一致", {}
    lo, hi = _num(time_min), _num(time_max)
    if lo is None or hi is None:
        return False, "range_invalid", "起止时点不是有效数字", {}
    if lo > hi:
        return False, "range_invalid", "时间范围倒序（min > max）", {}
    if hi > now + RANGE_TOLERANCE_SECONDS:
        return False, "range_invalid", "时间范围进入未来", {}
    live_lo, live_hi = _num(live.get("time_min")), _num(live.get("time_max"))
    if live_lo is None or live_hi is None:
        return False, "range_invalid", "当前导入没有可核对的时间范围", {}
    if lo < live_lo - RANGE_TOLERANCE_SECONDS or hi > live_hi + RANGE_TOLERANCE_SECONDS:
        return False, "range_invalid", "时间范围超出当前导入", {}
    if complete and (lo > live_lo + RANGE_TOLERANCE_SECONDS
                     or hi < live_hi - RANGE_TOLERANCE_SECONDS):
        # complete=True claims the WHOLE processed source; a narrow interior slice
        # (e.g. only the first conversation while all counts are total) is not full.
        return False, "range_invalid", "complete 声明的范围未覆盖已导入的全部有日期消息", {}
    normalized = {
        "processed": {"messages": msgs, "conversations": convs},
        "time_min": lo, "time_max": hi,
        "time_min_text": _date_text(lo), "time_max_text": _date_text(hi),
    }
    return True, "", "", normalized


def _source_binding(marker: dict[str, Any], latest: dict[str, Any] | None) -> tuple[bool, str, str]:
    if latest is None:
        return False, "no_successful_import", ""
    stored_db = latest.get("source_db_sha256")
    stored_man = latest.get("manifest_sha256")
    if not stored_db:
        return False, "source_unbound", ""
    if marker.get("source_db_sha256") != stored_db:
        return False, "source_mismatch", ""
    if stored_man:
        # Stored manifest hash exists -> the marker MUST carry it and match exactly.
        # A marker that is missing the field is rejected (no "absent field bypass").
        if marker.get("manifest_sha256") != stored_man:
            return False, "source_mismatch", ""
    elif marker.get("manifest_sha256"):
        # Only a legacy metadata row (imported before the manifest hash was recorded)
        # has no stored manifest hash; a marker that nonetheless carries one cannot be
        # bound to anything -> refuse rather than imply a verification that never ran.
        return False, "source_mismatch", ""
    return True, "", "ok"


def _user_state_dir(state_dir: Path | str, user_id: str) -> Path | None:
    """The state dir of *this* user, matched by the manifest's own ``user_id``."""
    users_dir = Path(state_dir) / "users"
    if not users_dir.exists():
        return None
    for path in sorted(users_dir.glob("*/manifest.json")):
        try:
            data = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("user_id") == user_id:
            return path.parent
    return None


def _historical_summary_hashes(user_dir: Path | None, version: int) -> set[str]:
    """sha256 of every IMMUTABLE per-version ``summary.md`` for ``version``.

    The engine publishes each version into its own directory (``versions/v<N>`` or
    the engine's ``versions/v<N>-<uuid>``) and never overwrites a published
    snapshot, so the content is an immutable record of what that version said. Only
    directories inside this user's own state dir are read.
    """
    if user_dir is None:
        return set()
    vroot = Path(user_dir) / "versions"
    if not vroot.exists():
        return set()
    hashes: set[str] = set()
    for cand in sorted(vroot.glob(f"v{version}")) + sorted(vroot.glob(f"v{version}-*")):
        try:
            text = (cand / "summary.md").read_text("utf-8")
        except OSError:
            continue
        hashes.add(_sha256_text(text))
    return hashes


def _summary_gate(marker: dict[str, Any], published: Published,
                  user_dir: Path | None) -> tuple[bool, str, int | None]:
    version = marker.get("memory_version")
    if not _is_int(version) or version <= 0:
        return False, "marker_unreadable", None
    if not published.present:
        return False, "no_published_summary", version
    if published.version < version:
        # the marker claims a version that was never published -> stale/forward marker
        return False, "summary_version_stale", version
    recorded = marker.get("memory_summary_sha256")
    if published.version == version:
        if not recorded:
            return False, "summary_hash_missing", version
        if recorded != _sha256_text(published.text):
            return False, "summary_hash_mismatch", version
        return True, "", version
    # published.version > version: memory has advanced past the marker. The claim
    # still holds ONLY if the marker's version really was published for this user
    # with exactly the recorded content (an immutable per-version release). A mere
    # larger current version number proves nothing about the old record.
    hashes = _historical_summary_hashes(user_dir, version)
    if not hashes:
        return False, "history_version_missing", version
    if not recorded:
        # an absent recorded hash is never accepted, at the same OR a higher version
        return False, "summary_hash_missing", version
    if recorded not in hashes:
        return False, "summary_hash_mismatch", version
    return True, "", version


def _load_marker(path: Path) -> tuple[dict[str, Any] | None, str]:
    if not path.exists():
        return None, "marker_missing"
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "marker_unreadable"
    if not isinstance(data, dict):
        return None, "marker_unreadable"
    if data.get("schema") != SCHEMA:
        return None, "marker_schema_mismatch"
    return data, ""


def _result(state: str, reason: str, *, path: Path, marker: dict[str, Any] | None = None,
            normalized: dict[str, Any] | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "state": state,
        "reason": reason,
        "reason_text": REASON_TEXT.get(reason, reason),
        "marker_path": str(path),
        "marker": marker,
        "range": None,
        "range_text": None,
        "complete": (marker or {}).get("complete"),
        "memory_version": (marker or {}).get("memory_version"),
        "processed": (marker or {}).get("processed"),
        "written_at": (marker or {}).get("written_at"),
    }
    if normalized:
        out["range"] = {"min": normalized["time_min"], "max": normalized["time_max"],
                        "min_text": normalized["time_min_text"], "max_text": normalized["time_max_text"]}
        out["range_text"] = f"{normalized['time_min_text']} ~ {normalized['time_max_text']}"
        out["processed"] = normalized["processed"]
    if extra:
        out.update(extra)
    if state == COVERED:
        out["label"] = f"常驻摘要已处理的历史范围 {out['range_text']}"
    elif state == PARTIAL:
        out["label"] = f"常驻摘要已处理的历史范围（部分完成）{out['range_text']}"
    else:
        out["label"] = f"未知（{out['reason_text']}）"
    return out


def compute_coverage(config: Config, *, live: dict[str, Any], latest: dict[str, Any] | None,
                     published: Published, state_dir: Path | str | None = None,
                     marker: Path | str | None = None,
                     now: float | None = None) -> dict[str, Any]:
    """Evaluate the marker for display. Read-only; never raises, never writes."""
    path = marker_path(config, marker)
    state = Path(state_dir) if state_dir is not None else Path(config.maint_state_dir)
    now = float(now) if now is not None else time.time()
    try:
        data, reason = _load_marker(path)
        if data is None:
            return _result(UNKNOWN, reason, path=path)
        if data.get("user_id") != config.user_id:
            return _result(UNKNOWN, "marker_user_mismatch", path=path, marker=data)
        bound, reason, _ = _source_binding(data, latest)
        if not bound:
            return _result(UNKNOWN, reason, path=path, marker=data)
        ok, reason, version = _summary_gate(
            data, published, _user_state_dir(state, config.user_id))
        if not ok:
            return _result(UNKNOWN, reason, path=path, marker=data)
        ok, reason, _text, normalized = _validate_claims(
            processed=data.get("processed"), time_min=data.get("time_min"),
            time_max=data.get("time_max"), complete=data.get("complete"), live=live, now=now)
        if not ok:
            return _result(UNKNOWN, reason, path=path, marker=data)
        state = COVERED if data.get("complete") is True else PARTIAL
        return _result(state, "ok", path=path, marker=data, normalized=normalized,
                       extra={"memory_version": version, "source_id": latest.get("source_id")})
    except Exception as exc:  # noqa: BLE001 - coverage must never break status/聊天
        return _result(UNKNOWN, "coverage_error", path=path, extra={"error": f"{type(exc).__name__}: {exc}"})


# --------------------------------------------------------------------------- #
# Atomic writer (main Agent's bootstrap calls this ONLY after a real publish)
# --------------------------------------------------------------------------- #
def write_coverage_marker(config: Config, *, memory_version: int,
                          processed_messages: int, processed_conversations: int,
                          time_min: float, time_max: float, complete: bool,
                          source_db_sha256: str | None = None,
                          manifest_sha256: str | None = None,
                          conn: sqlite3.Connection | None = None,
                          state_dir: Path | str | None = None,
                          marker: Path | str | None = None,
                          now: float | None = None) -> dict[str, Any]:
    """Write the marker atomically, after verifying the claim against real state.

    Everything is checked before a byte is written: a successful import row must
    exist, the source hashes must equal the STORED ones, counts/time must be valid
    for the live import (``complete=True`` must cover the FULL dated range of the
    import), and ``memory_version`` must be the version actually published for this
    user AND present as an immutable per-version release whose content matches the
    published summary (the real summary hash is recorded, never supplied by the
    caller). On any refusal/failure nothing is written (a previous marker is
    left as-is).
    """
    marker_file = marker_path(config, marker)
    now = float(now) if now is not None else time.time()
    state = Path(state_dir) if state_dir is not None else Path(config.maint_state_dir)
    own = conn is None
    if own:
        conn = db.connect(config.db_path)
        db.init_db(conn)
    try:
        latest = latest_ok_import(conn)
        if latest is None:
            return {"status": "failed", "reason": "no_successful_import",
                    "error": REASON_TEXT["no_successful_import"]}
        stored_db = latest.get("source_db_sha256")
        stored_man = latest.get("manifest_sha256")
        if not stored_db:
            return {"status": "failed", "reason": "source_unbound",
                    "error": REASON_TEXT["source_unbound"]}
        if source_db_sha256 is not None and source_db_sha256 != stored_db:
            return {"status": "failed", "reason": "source_mismatch",
                    "error": "提交的来源 hash 与当前成功导入的 stored source_db_sha256 不符"}
        if manifest_sha256 is not None:
            if not stored_man:
                return {"status": "failed", "reason": "source_unbound",
                        "error": "当前成功导入元数据没有 manifest hash，无法核对提交的 manifest hash"}
            if manifest_sha256 != stored_man:
                return {"status": "failed", "reason": "source_mismatch",
                        "error": "提交的 manifest hash 与当前成功导入不符"}

        live = _live_stats(conn)
        ok, reason, text, normalized = _validate_claims(
            processed={"messages": processed_messages, "conversations": processed_conversations},
            time_min=time_min, time_max=time_max, complete=complete, live=live, now=now)
        if not ok:
            return {"status": "failed", "reason": reason, "error": text or REASON_TEXT.get(reason, reason)}

        published = read_published(state, config.user_id)
        if not published.present or published.version != memory_version:
            return {"status": "failed", "reason": "version_not_published",
                    "error": ("memory_version 必须等于当前真实已发布版本 "
                              f"（已发布 {published.version if published.present else 'none'}，"
                              f"提交 {memory_version}）"),
                    "published_version": published.version if published.present else 0}
        # Bind the recorded hash to the version's IMMUTABLE release: the same check
        # the reader will apply once a later version is published. A published
        # manifest with no immutable per-version record cannot be claimed.
        published_sha = _sha256_text(published.text)
        release_hashes = _historical_summary_hashes(
            _user_state_dir(state, config.user_id), memory_version)
        if not release_hashes:
            return {"status": "failed", "reason": "history_version_missing",
                    "error": REASON_TEXT["history_version_missing"]}
        if published_sha not in release_hashes:
            return {"status": "failed", "reason": "summary_hash_mismatch",
                    "error": "当前已发布摘要与该版本不可变发布记录的内容不一致"}

        payload = {
            "schema": SCHEMA,
            "user_id": config.user_id,
            "source_id": latest.get("source_id"),
            "source_db_sha256": stored_db,
            "manifest_sha256": stored_man,
            "processed": normalized["processed"],
            "time_min": normalized["time_min"],
            "time_max": normalized["time_max"],
            "time_min_text": normalized["time_min_text"],
            "time_max_text": normalized["time_max_text"],
            "complete": complete,
            "memory_version": memory_version,
            "memory_summary_sha256": published_sha,
            "written_at": _utc_now(),
            "note": ("本标记只声明已处理的源范围；不等于“记住了全部细节”，"
                     "也不代表原始来源被重导或改写。"),
        }
        try:
            _atomic_write_json(marker_file, payload)
        except OSError as exc:
            return {"status": "failed", "reason": "write_failed",
                    "error": f"原子写入标记失败：{type(exc).__name__}: {exc}",
                    "marker_path": str(marker_file)}
        return {"status": "ok", "marker_path": str(marker_file), "marker": payload,
                "source_id": latest.get("source_id"), "memory_version": memory_version}
    finally:
        if own:
            conn.close()
