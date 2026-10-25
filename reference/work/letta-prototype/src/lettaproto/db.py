"""Canonical SQLite store. Independent of Letta; exportable."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS source_files (
  path TEXT PRIMARY KEY,
  size INTEGER,
  sha256 TEXT,
  mtime REAL,
  kind TEXT
);

CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY,
  title TEXT,
  create_time REAL,
  update_time REAL,
  current_node TEXT,
  is_archived INTEGER,
  is_starred INTEGER,
  model_slug TEXT,
  source_file TEXT,
  node_count INTEGER,
  message_count INTEGER
);

CREATE TABLE IF NOT EXISTS nodes (
  conversation_id TEXT,
  node_id TEXT,
  parent_id TEXT,
  child_count INTEGER,
  is_branch_point INTEGER,
  is_selected_branch INTEGER,
  has_message INTEGER,
  PRIMARY KEY (conversation_id, node_id)
);

CREATE TABLE IF NOT EXISTS messages (
  conversation_id TEXT,
  node_id TEXT,
  message_id TEXT,
  parent_id TEXT,
  role TEXT,
  content_type TEXT,
  text TEXT,
  create_time REAL,
  is_selected_branch INTEGER,
  is_assistant_reasoning INTEGER,
  has_attachment INTEGER,
  PRIMARY KEY (conversation_id, node_id)
);

CREATE TABLE IF NOT EXISTS branches (
  conversation_id TEXT,
  parent_node_id TEXT,
  child_count INTEGER,
  PRIMARY KEY (conversation_id, parent_node_id)
);

CREATE TABLE IF NOT EXISTS attachments (
  conversation_id TEXT,
  node_id TEXT,
  message_id TEXT,
  file_id TEXT,
  name TEXT,
  mime_type TEXT,
  size INTEGER,
  width INTEGER,
  height INTEGER,
  asset_pointer TEXT,
  local_dat TEXT,
  PRIMARY KEY (conversation_id, node_id, file_id)
);

CREATE TABLE IF NOT EXISTS import_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  started_at TEXT,
  finished_at TEXT,
  source_signature TEXT,
  conversations INTEGER,
  nodes INTEGER,
  messages INTEGER,
  status TEXT
);

CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_msg_role_time ON messages(role, create_time);
CREATE INDEX IF NOT EXISTS idx_msg_sel ON messages(conversation_id, is_selected_branch);

-- working-memory (idea.md) version history; each save is one row + a snapshot file
CREATE TABLE IF NOT EXISTS idea_versions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  instance TEXT,
  version INTEGER,
  created_at TEXT,
  note TEXT,
  sha256 TEXT,
  chars INTEGER,
  path TEXT
);

-- incremental updates to the working memory, with provenance + rollback target
CREATE TABLE IF NOT EXISTS memory_updates (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT,
  instance TEXT,
  trigger TEXT,
  prev_version INTEGER,
  new_version INTEGER,
  diff_added INTEGER,
  diff_removed INTEGER,
  sources_json TEXT,
  summary TEXT,
  status TEXT
);

-- prototype chat turns (separate from the imported history)
CREATE TABLE IF NOT EXISTS chat_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at REAL,
  instance TEXT,
  conversation_id TEXT,
  role TEXT,
  content TEXT,
  sources_json TEXT
);

-- background memory tasks (update suggestions + idle increments), persisted queue
CREATE TABLE IF NOT EXISTS memory_tasks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  dedup_key TEXT UNIQUE,
  created_at TEXT,
  updated_at TEXT,
  instance TEXT,
  kind TEXT,              -- suggestion | idle | correction
  status TEXT,            -- submitted | processing | saved | failed
  spec_json TEXT,
  source_ids_json TEXT,
  attempts INTEGER DEFAULT 0,
  result_json TEXT,
  error TEXT,
  started_at TEXT,
  finished_at TEXT
);

-- observability for the worker's search/collaboration traces
CREATE TABLE IF NOT EXISTS search_traces (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT,
  instance TEXT,
  query TEXT,
  rounds INTEGER,
  status TEXT,
  sources_json TEXT,
  result_json TEXT
);
"""


def connect(db_path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def get_meta(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
