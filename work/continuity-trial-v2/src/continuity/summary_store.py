"""Read the *published* long-term summary produced by the maintenance CLI.

The maintenance engine publishes atomically: ``manifest.json`` is written LAST
and is the single pointer to a complete version. This module only ever reads the
manifest and the file it points at. It never reads a version directory directly,
so a half-written or failed batch can never be surfaced: a failure leaves the old
complete summary in place, and the next chat turn keeps using it.

If no version has ever been published, the trial reports an explicit EMPTY memory
and says so, instead of pretending to have read the user's history.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

EMPTY_MEMORY_TEXT = (
    "# 常驻摘要（尚未形成）\n\n"
    "本试用还没有任何已核验的长期记忆。以下内容不属于你的历史，不要假装读过你的个人资料；"
    "只依据当前会话和你在本试用中说过的话作答。\n"
)


def empty_summary_hint() -> str:
    return "当前没有任何已发布的长期记忆；回答只依据本会话与试用内的对话，不假装读过历史。"


@dataclass
class Published:
    present: bool
    version: int = 0
    text: str = ""
    published_at: str | None = None
    now_max: str | None = None
    summary_path: str | None = None
    memory_path: str | None = None
    user_id: str | None = None
    reason: str = ""

    @property
    def display_text(self) -> str:
        return self.text if self.present else EMPTY_MEMORY_TEXT


def _load_manifest(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _select_manifest(state_dir: Path, user_id: str | None) -> tuple[Path, dict] | Published:
    """Return (path, manifest) for the requested user, or a fail-safe Published.

    Strict: a named user must match a manifest's own ``user_id``; we never fall back
    to another user's data. Without a user_id, >1 user is ambiguous and refused.
    """
    users_dir = state_dir / "users"
    if not users_dir.exists():
        return Published(present=False, reason="no published version yet")
    manifests = sorted(users_dir.glob("*/manifest.json"))
    if not manifests:
        return Published(present=False, reason="no published version yet")
    if user_id:
        for path in manifests:
            data = _load_manifest(path)
            if data and data.get("user_id") == user_id:
                return path, data
        return Published(present=False, user_id=user_id,
                         reason=f"no manifest for user {user_id!r}; refusing to read another user")
    if len(manifests) > 1:
        return Published(present=False,
                         reason=f"{len(manifests)} users present; user_id is required")
    data = _load_manifest(manifests[0])
    if data is None:
        return Published(present=False, reason="manifest unreadable")
    return manifests[0], data


def _within(base: Path, target: Path) -> bool:
    try:
        target.relative_to(base)
        return True
    except ValueError:
        return False


def read_published(state_dir: Path | str, user_id: str | None = None) -> Published:
    state_dir = Path(state_dir)
    selected = _select_manifest(state_dir, user_id)
    if isinstance(selected, Published):
        return selected
    _manifest_path, manifest = selected
    manifest_user = manifest.get("user_id")
    if user_id and manifest_user != user_id:
        # defensive: a manifest that names a different user must never be read
        return Published(present=False, user_id=user_id,
                         reason="manifest user mismatch; refusing to read another user")

    summary_path = manifest.get("summary_path")
    if not summary_path:
        return Published(present=False, reason="manifest has no summary_path")
    version = int(manifest.get("version") or 0)
    if version <= 0:
        # version 0 is the engine's pre-claim placeholder (it also publishes one on
        # a fail-closed first batch). Treat it as "no memory yet", not as memory.
        return Published(present=False, version=0, user_id=manifest_user,
                         reason="no claims yet (engine version 0 placeholder)")
    path = Path(summary_path).resolve()
    if not _within(state_dir.resolve(), path):
        return Published(present=False, version=version, user_id=manifest_user,
                         reason=f"summary_path escapes the state dir; refusing to read {path}")
    try:
        text = path.read_text("utf-8")
    except OSError:
        # manifest points at a missing/corrupt file: fail safe to "empty", never
        # fabricate memory. The engine's reconcile() will republish on next batch.
        return Published(present=False, version=version, user_id=manifest_user,
                         reason=f"published summary missing: {path}")
    expected = manifest.get("summary_sha")
    if not expected:
        return Published(present=False, version=version, user_id=manifest_user,
                         reason="manifest has no summary_sha; unverifiable, refusing to inject")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != expected:
        return Published(present=False, version=version, user_id=manifest_user,
                         reason="summary hash mismatch (corrupted or half-written); refusing to inject")
    if not text.strip():
        return Published(present=False, version=version, user_id=manifest_user,
                         reason="published summary empty")
    return Published(
        present=True,
        version=version,
        text=text,
        published_at=manifest.get("published_at"),
        now_max=manifest.get("now_max"),
        summary_path=str(path),
        memory_path=manifest.get("memory_path"),
        user_id=manifest_user,
    )
