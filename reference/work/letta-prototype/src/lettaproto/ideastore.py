"""Working memory = the user's chosen `idea.md` snapshot, versioned + rollback-able.

Two instances:
  prod  -> a read-only copy of the user's source idea.md (the real working memory)
  test  -> a separate, explicitly marked copy used for mutation demos/tests

The user's ORIGINAL file is never modified. Every save is atomic and produces a
snapshot under <instance>/versions/, so a failure or a bad update can be rolled
back without losing the previous working copy.
"""

from __future__ import annotations

import difflib
import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import db
from .config import Config

TEST_MARKER = "<!-- TEST INSTANCE: copy of idea.md for prototype demos; not the production working memory -->"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def idea_path(config: Config, instance: str = "prod") -> Path:
    return config.memory_root / instance / "idea.md"


def versions_dir(config: Config, instance: str = "prod") -> Path:
    return config.memory_root / instance / "versions"


def load(config: Config, instance: str = "prod") -> str:
    path = idea_path(config, instance)
    if not path.exists():
        raise FileNotFoundError(f"working memory not initialized: {path}")
    return path.read_text("utf-8")


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, "utf-8")
    tmp.replace(path)  # atomic; on failure the original file is untouched


def current_version(conn: sqlite3.Connection, instance: str) -> int:
    row = conn.execute(
        "SELECT MAX(version) v FROM idea_versions WHERE instance=?", (instance,)
    ).fetchone()
    return int(row["v"] or 0)


def seed_prod(config: Config, conn: sqlite3.Connection) -> dict[str, Any]:
    """Copy the user's source idea.md into the working prod instance once."""
    target = idea_path(config, "prod")
    if target.exists():
        return {"instance": "prod", "action": "kept", "version": current_version(conn, "prod")}
    text = config.source_idea.read_text("utf-8")
    return save(config, conn, text, note=f"seed from {config.source_idea.name}", instance="prod", source_sha=hashlib.sha256(text.encode()).hexdigest())


def seed_test(config: Config, conn: sqlite3.Connection) -> dict[str, Any]:
    """Create a separate, explicitly marked test instance copied from the source."""
    target = idea_path(config, "test")
    if target.exists():
        return {"instance": "test", "action": "kept", "version": current_version(conn, "test")}
    text = config.source_idea.read_text("utf-8")
    marked = TEST_MARKER + "\n\n" + text
    return save(config, conn, marked, note="seed test instance (marked copy)", instance="test")


def save(
    config: Config,
    conn: sqlite3.Connection,
    text: str,
    *,
    note: str = "",
    instance: str = "prod",
    source_sha: str | None = None,
) -> dict[str, Any]:
    db.init_db(conn)
    version = current_version(conn, instance) + 1
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    vdir = versions_dir(config, instance)
    vdir.mkdir(parents=True, exist_ok=True)
    snap = vdir / f"idea-v{version:04d}.md"
    snap.write_text(text, "utf-8")
    _write_atomic(idea_path(config, instance), text)
    conn.execute(
        "INSERT INTO idea_versions(instance, version, created_at, note, sha256, chars, path) VALUES(?,?,?,?,?,?,?)",
        (instance, version, _now(), note, digest, len(text), str(snap)),
    )
    conn.commit()
    return {
        "instance": instance,
        "version": version,
        "sha256": digest,
        "chars": len(text),
        "path": str(idea_path(config, instance)),
        "snapshot": str(snap),
        "source_sha": source_sha,
    }


def diff_stat(old: str, new: str) -> dict[str, int]:
    added = removed = 0
    for line in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm=""):
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return {"added": added, "removed": removed}


def unified_diff(old: str, new: str, context: int = 2) -> str:
    return "\n".join(
        difflib.unified_diff(old.splitlines(), new.splitlines(), "idea.prev", "idea.new", lineterm="", n=context)
    )


def list_versions(conn: sqlite3.Connection, instance: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT version, created_at, note, chars, sha256, path FROM idea_versions WHERE instance=? ORDER BY version",
        (instance,),
    ).fetchall()
    return [dict(r) for r in rows]


def rollback(config: Config, conn: sqlite3.Connection, instance: str, version: int) -> dict[str, Any]:
    row = conn.execute(
        "SELECT path FROM idea_versions WHERE instance=? AND version=?", (instance, version)
    ).fetchone()
    if not row:
        raise ValueError(f"version {version} not found for instance {instance}")
    text = Path(row["path"]).read_text("utf-8")
    info = save(config, conn, text, note=f"rollback to v{version}", instance=instance)
    info["rolled_back_to"] = version
    return info


def stats(config: Config, conn: sqlite3.Connection, instance: str) -> dict[str, Any]:
    from .util import approx_tokens

    try:
        text = load(config, instance)
    except FileNotFoundError:
        return {"instance": instance, "initialized": False}
    return {
        "instance": instance,
        "initialized": True,
        "path": str(idea_path(config, instance)),
        "version": current_version(conn, instance),
        "chars": len(text),
        "lines": text.count("\n") + 1,
        "token_estimate": approx_tokens(text),
        "sha256_prefix": hashlib.sha256(text.encode()).hexdigest()[:12],
        "is_test_marked": TEST_MARKER in text,
    }
