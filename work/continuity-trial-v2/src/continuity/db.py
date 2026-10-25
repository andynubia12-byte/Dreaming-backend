"""Isolated SQLite store for the continuity trial.

Schema is a deliberate subset of the 8788 ``lettaproto/db.py`` schema
(sha256 2024b207d70d7ca7174409c119f392caa76c61ab84a2661284d383b0bb13511c):

* ``messages`` / ``conversations`` — imported export history, filled ONLY by the
  explicit read-only import in ``history.py`` (schema-compatible with
  ``history-bootstrap-v1``'s ``retrieval_import.sqlite``). The retrieval agent
  (``retrieval.py``) reads ``messages`` unchanged; imported history is NEVER
  written into ``chat_log``.
* ``chat_log`` — this trial's own turn store (one row per user/assistant message).
* ``maint_queue`` — the durable serial maintenance queue (trial-specific).
* ``history_imports`` — independent metadata about each import attempt (source
  hashes, counts, time range, and an explicit "summary coverage unknown" field).

Nothing here points at the live 8788 database; the trial has its own file.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
-- imported export history (left empty by this trial; table kept for the copied
-- retrieval agent). Columns match lettaproto/messages exactly.
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
CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_msg_sel ON messages(conversation_id, is_selected_branch);

-- imported conversations (history-bootstrap-v1 retrieval schema)
CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY, title TEXT, create_time REAL, update_time REAL, current_node TEXT,
  node_count INTEGER, message_count INTEGER, source_file TEXT
);

-- independent import metadata: source hashes, live counts, time range, coverage.
CREATE TABLE IF NOT EXISTS history_imports (
  source_id TEXT PRIMARY KEY,
  status TEXT,                 -- ok | failed
  source_db TEXT, source_db_sha256 TEXT,
  manifest TEXT, manifest_sha256 TEXT,
  imported_at TEXT,
  messages INTEGER, conversations INTEGER, users INTEGER, assistants INTEGER,
  skipped INTEGER, undated INTEGER,
  time_min REAL, time_max REAL,
  summary_covered_range TEXT,  -- always 'unknown': we never claim full coverage
  error TEXT, note TEXT
);

-- this trial's own chat turns (separate from any imported history)
CREATE TABLE IF NOT EXISTS chat_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at REAL,
  instance TEXT,
  conversation_id TEXT,
  role TEXT,
  content TEXT,
  sources_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_chatlog_conv ON chat_log(instance, conversation_id, id);

-- durable, serial maintenance queue (one row per successful chat turn)
CREATE TABLE IF NOT EXISTS maint_queue (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  batch_id TEXT UNIQUE,
  kind TEXT,
  status TEXT,              -- pending | processing | done | failed
  created_at REAL,
  updated_at REAL,
  started_at REAL,
  finished_at REAL,
  attempts INTEGER DEFAULT 0,
  user_id TEXT,
  payload_json TEXT,        -- the exact batch handed to the maintenance CLI
  result_json TEXT,
  error TEXT,
  summary_version INTEGER
);
CREATE INDEX IF NOT EXISTS idx_queue_status ON maint_queue(status, id);
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
