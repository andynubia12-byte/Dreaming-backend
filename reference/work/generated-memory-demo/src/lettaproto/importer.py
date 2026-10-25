"""Import a ChatGPT export into the canonical store (idempotent).

Design rules enforced here:
  * preserve conversation / node / message / source IDs
  * derive the selected branch from `current_node` (this export has no children
    field, so children are derived from parent pointers)
  * do NOT concatenate branches into facts; only mark them
  * assistant `thoughts` / `reasoning_recap` are stored but flagged as reasoning
    (never treated as user facts)
  * attachments are indexed only (no binary upload)
"""

from __future__ import annotations

import glob
import json
import sqlite3
from pathlib import Path
from typing import Any

from . import db
from .util import approx_tokens, sha256_file, text_from_parts, write_json

REASONING_CONTENT_TYPES = {"thoughts", "reasoning_recap"}


def _parse_assets(message: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    content = message.get("content") or {}
    for part in content.get("parts") or []:
        if isinstance(part, dict) and part.get("asset_pointer"):
            out.append(
                {
                    "file_id": str(part.get("asset_pointer", "")).replace("sediment://", "").split("/")[-1],
                    "asset_pointer": part.get("asset_pointer"),
                    "name": None,
                    "mime_type": part.get("content_type"),
                    "size": part.get("size_bytes"),
                    "width": part.get("width"),
                    "height": part.get("height"),
                }
            )
    for att in (message.get("metadata") or {}).get("attachments") or []:
        out.append(
            {
                "file_id": att.get("id"),
                "asset_pointer": None,
                "name": att.get("name"),
                "mime_type": att.get("mime_type"),
                "size": att.get("size"),
                "width": att.get("width"),
                "height": att.get("height"),
            }
        )
    for asset in content.get("assets") or []:
        if isinstance(asset, dict) and asset.get("asset_pointer"):
            out.append(
                {
                    "file_id": str(asset.get("asset_pointer", "")).replace("sediment://", "").split("/")[-1],
                    "asset_pointer": asset.get("asset_pointer"),
                    "name": asset.get("name"),
                    "mime_type": asset.get("mime_type"),
                    "size": asset.get("size"),
                    "width": asset.get("width"),
                    "height": asset.get("height"),
                }
            )
    return [a for a in out if a.get("file_id")]


def _selected_path(mapping: dict[str, Any], current_node: str | None) -> set[str]:
    """Walk parents from current_node to root; the selected branch."""
    path: set[str] = set()
    seen: set[str] = set()
    node_id = current_node
    while node_id and node_id in mapping and node_id not in seen:
        seen.add(node_id)
        path.add(node_id)
        node_id = (mapping.get(node_id) or {}).get("parent")
    return path


def scan_source(export_dir: Path) -> list[dict[str, Any]]:
    """Manifest + checksums of source export files (read-only; files untouched)."""
    files: list[dict[str, Any]] = []
    patterns = ["conversations-*.json", "*.json", "*.md"]
    seen: set[Path] = set()
    for pattern in patterns:
        for raw in sorted(glob.glob(str(export_dir / pattern))):
            path = Path(raw)
            if path in seen or path.name.startswith("."):
                continue
            seen.add(path)
            files.append(
                {
                    "path": str(path),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "mtime": path.stat().st_mtime,
                    "kind": "conversations" if path.name.startswith("conversations-") else path.suffix.lstrip("."),
                }
            )
    return files


def _load_asset_names(export_dir: Path) -> dict[str, str]:
    path = export_dir / "conversation_asset_file_names.json"
    if not path.exists():
        return {}
    return {k: v for k, v in json.loads(path.read_text("utf-8")).items()}


def _import_conversation(
    conn: sqlite3.Connection,
    conv: dict[str, Any],
    source_file: str,
    asset_names: dict[str, str],
) -> tuple[int, int, int]:
    cid = conv.get("id") or conv.get("conversation_id")
    mapping: dict[str, Any] = conv.get("mapping") or {}
    current_node = conv.get("current_node")
    selected = _selected_path(mapping, current_node)

    child_count: dict[str, int] = {}
    for node in mapping.values():
        parent = node.get("parent")
        if parent is not None:
            child_count[parent] = child_count.get(parent, 0) + 1

    # wipe this conversation's rows so re-import is idempotent (no duplicates)
    for table in ("nodes", "messages", "branches", "attachments"):
        conn.execute(f"DELETE FROM {table} WHERE conversation_id=?", (cid,))

    n_nodes = n_msgs = 0
    for node_id, node in mapping.items():
        message = node.get("message")
        parent_id = node.get("parent")
        cc = child_count.get(node_id, 0)
        is_selected = 1 if node_id in selected else 0
        conn.execute(
            "INSERT OR REPLACE INTO nodes VALUES(?,?,?,?,?,?,?)",
            (cid, node_id, parent_id, cc, 1 if cc > 1 else 0, is_selected, 1 if message else 0),
        )
        n_nodes += 1
        if not message:
            continue
        role = (message.get("author") or {}).get("role") or "unknown"
        content = message.get("content") or {}
        ctype = content.get("content_type") or "text"
        text = text_from_parts(content.get("parts") or [])
        assets = _parse_assets(message)
        reasoning = 1 if ctype in REASONING_CONTENT_TYPES else 0
        conn.execute(
            "INSERT OR REPLACE INTO messages VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                cid,
                node_id,
                message.get("id"),
                parent_id,
                role,
                ctype,
                text,
                message.get("create_time"),
                is_selected,
                reasoning,
                1 if assets else 0,
            ),
        )
        n_msgs += 1
        for asset in assets:
            fid = asset["file_id"]
            conn.execute(
                "INSERT OR REPLACE INTO attachments VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    cid,
                    node_id,
                    message.get("id"),
                    fid,
                    asset.get("name") or asset_names.get(f"{fid}.dat"),
                    asset.get("mime_type"),
                    asset.get("size"),
                    asset.get("width"),
                    asset.get("height"),
                    asset.get("asset_pointer"),
                    f"{fid}.dat" if f"{fid}.dat" in asset_names else None,
                ),
            )

    for parent_node_id, cc in child_count.items():
        if cc > 1:
            conn.execute(
                "INSERT OR REPLACE INTO branches VALUES(?,?,?)", (cid, parent_node_id, cc)
            )

    conn.execute(
        "INSERT OR REPLACE INTO conversations VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (
            cid,
            conv.get("title"),
            conv.get("create_time"),
            conv.get("update_time"),
            current_node,
            1 if conv.get("is_archived") else 0,
            1 if conv.get("is_starred") else 0,
            conv.get("default_model_slug"),
            source_file,
            n_nodes,
            n_msgs,
        ),
    )
    return n_nodes, n_msgs, len(selected)


def run_import(export_dir: Path, db_path: Path, report_path: Path | None = None) -> dict[str, Any]:
    conn = db.connect(db_path)
    db.init_db(conn)
    source_files = scan_source(export_dir)
    for sf in source_files:
        conn.execute(
            "INSERT OR REPLACE INTO source_files VALUES(?,?,?,?,?)",
            (sf["path"], sf["size"], sf["sha256"], sf["mtime"], sf["kind"]),
        )
    asset_names = _load_asset_names(export_dir)

    started = db.get_meta(conn, "import_count")
    conn.execute(
        "INSERT INTO import_runs(started_at, source_signature, status) VALUES(datetime('now'), ?, 'running')",
        (str(len(source_files)),),
    )
    run_id = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]

    total_conv = total_nodes = total_msgs = 0
    conv_files = sorted(glob.glob(str(export_dir / "conversations-*.json")))
    for path_str in conv_files:
        path = Path(path_str)
        data = json.loads(path.read_text("utf-8"))
        for conv in data:
            n, m, _ = _import_conversation(conn, conv, path.name, asset_names)
            total_conv += 1
            total_nodes += n
            total_msgs += m
    conn.execute(
        "UPDATE import_runs SET finished_at=datetime('now'), conversations=?, nodes=?, messages=?, status='ok' WHERE id=?",
        (total_conv, total_nodes, total_msgs, run_id),
    )
    db.set_meta(conn, "import_count", str(int(started or 0) + 1))
    conn.commit()

    report = {
        "conversations": total_conv,
        "nodes": total_nodes,
        "messages": total_msgs,
        "source_files": len(source_files),
        "conv_files": len(conv_files),
        "db_path": str(db_path),
    }
    if report_path:
        write_json(report_path, report)
    conn.close()
    return report


def user_message_index(conn: sqlite3.Connection, selected_only: bool = True) -> list[sqlite3.Row]:
    sql = (
        "SELECT m.node_id, m.conversation_id, m.message_id, m.create_time, m.text, c.title "
        "FROM messages m JOIN conversations c ON c.id=m.conversation_id "
        "WHERE m.role='user' AND m.is_assistant_reasoning=0"
    )
    if selected_only:
        sql += " AND m.is_selected_branch=1"
    sql += " ORDER BY m.create_time"
    return conn.execute(sql).fetchall()


def stats(conn: sqlite3.Connection) -> dict[str, Any]:
    q = lambda sql, *a: conn.execute(sql, a).fetchone()[0]  # noqa: E731
    return {
        "conversations": q("SELECT COUNT(*) FROM conversations"),
        "nodes": q("SELECT COUNT(*) FROM nodes"),
        "messages": q("SELECT COUNT(*) FROM messages"),
        "user_messages": q("SELECT COUNT(*) FROM messages WHERE role='user'"),
        "assistant_messages": q("SELECT COUNT(*) FROM messages WHERE role='assistant'"),
        "reasoning_messages": q("SELECT COUNT(*) FROM messages WHERE is_assistant_reasoning=1"),
        "selected_branch_messages": q("SELECT COUNT(*) FROM messages WHERE is_selected_branch=1"),
        "branched_conversations": q("SELECT COUNT(DISTINCT conversation_id) FROM branches"),
        "attachments": q("SELECT COUNT(*) FROM attachments"),
        "user_chars": q("SELECT COALESCE(SUM(LENGTH(text)),0) FROM messages WHERE role='user'"),
    }
