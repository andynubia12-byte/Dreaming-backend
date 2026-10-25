
-- Schema-only development copy, no data. Historical comment about empty tables
-- is not a statement about the old runtime. Columns match lettaproto/messages.
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
