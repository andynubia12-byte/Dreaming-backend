"""Transactional state store (SQLite) + atomic versioned publication.

The SQLite database is the single source of truth and holds the COMPLETE version
(including the summary and memory text). Publication is two-phase:

  1. each batch is applied and the full text committed inside ONE transaction;
  2. `publish(uid)` writes the versioned files (users/<uid>/versions/v<N>/) plus
     the cumulative evidence file, then writes manifest.json LAST.

manifest.json is the single atomic pointer to the published version. If any
derived write fails, the pointer still names the previous complete version, and
`reconcile(uid)` republishes the committed version from the database — so a crash
or write error can never leave a half-updated published version, and no version
is lost.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import hashlib
import json
import os
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Iterator

from .timeutil import SH, parse_flexible, parse_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  user_id TEXT PRIMARY KEY, created_at TEXT, now_max TEXT, version INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS batches(
  user_id TEXT, batch_id TEXT, applied_at TEXT, version INTEGER, status TEXT, note TEXT,
  fingerprint TEXT, PRIMARY KEY(user_id, batch_id));
CREATE TABLE IF NOT EXISTS messages(
  user_id TEXT, message_id TEXT, conversation_id TEXT, role TEXT, message_at TEXT,
  text TEXT, ingested_at TEXT, batch_id TEXT,
  PRIMARY KEY(user_id, message_id));
CREATE TABLE IF NOT EXISTS claims(
  claim_id TEXT PRIMARY KEY, user_id TEXT, text TEXT, kind TEXT, status TEXT,
  importance TEXT, valid_from TEXT, valid_to TEXT, relative_phrase TEXT, resolved_from_phrase TEXT,
  event_at TEXT, subject TEXT, first_seen TEXT, last_updated TEXT, created_version INTEGER,
  replaces TEXT, replaced_by TEXT, verification TEXT, note TEXT, edge TEXT);
CREATE TABLE IF NOT EXISTS claim_sources(
  claim_id TEXT, message_id TEXT, role TEXT, quote TEXT, message_at TEXT,
  PRIMARY KEY(claim_id, message_id, quote));
CREATE TABLE IF NOT EXISTS claim_events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, claim_id TEXT, user_id TEXT, at TEXT,
  kind TEXT, from_status TEXT, to_status TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS versions(
  user_id TEXT, version INTEGER, created_at TEXT, summary_sha TEXT, memory_sha TEXT,
  evidence_count INTEGER, claims_active INTEGER, note TEXT, summary_text TEXT, memory_text TEXT,
  rendered_now TEXT,
  PRIMARY KEY(user_id, version));
"""

SAFE = re.compile(r"[^A-Za-z0-9._-]")


def _now_iso() -> str:
    return _dt.datetime.now(SH).isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def dir_name(user_id: str) -> str:
    safe = SAFE.sub("_", user_id)[:48] or "user"
    return f"{safe}-{uuid.uuid5(uuid.NAMESPACE_URL, user_id).hex[:10]}"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, "utf-8")
    os.replace(tmp, path)


def new_claim_id() -> str:
    return "c-" + uuid.uuid4().hex[:10]


class Store:
    def __init__(self, state_dir: Path) -> None:
        self.state_dir = Path(state_dir)
        self.users_dir = self.state_dir / "users"
        self.evidence_dir = self.state_dir / "evidence"
        for d in (self.state_dir, self.users_dir, self.evidence_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.db_path = self.state_dir / "state.sqlite3"
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        def cols(table: str) -> set[str]:
            return {r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})")}
        if "fingerprint" not in cols("batches"):
            self.conn.execute("ALTER TABLE batches ADD COLUMN fingerprint TEXT")
        if "summary_text" not in cols("versions"):
            self.conn.execute("ALTER TABLE versions ADD COLUMN summary_text TEXT")
        if "memory_text" not in cols("versions"):
            self.conn.execute("ALTER TABLE versions ADD COLUMN memory_text TEXT")
        if "rendered_now" not in cols("versions"):
            self.conn.execute("ALTER TABLE versions ADD COLUMN rendered_now TEXT")
        if "edge" not in cols("claims"):
            self.conn.execute("ALTER TABLE claims ADD COLUMN edge TEXT")
        # Idempotent: only rows that still lack an edge state are touched, so this
        # is safe to run on every open (including states written by an older build).
        self._backfill_edges()

    def _backfill_edges(self) -> None:
        """Give pre-existing replacement claims a durable edge state.

        A claim created by this build marks an UNPROVEN replacement relation with a
        `created` event whose detail says `unproven` (and, for a single historical
        version, the same wording appeared in `note`). Those rows migrate to
        `edge='unknown'` so a later confirm cannot silently turn an unknown edge
        into a proven replacement. Every other legacy row that points at a target
        (`replaces` set) came from a build where a `correct`/`retract` implied an
        intended replacement, so it migrates to `edge='replaces'` — preserving the
        historical "confirm closes the weak-correction chain" behaviour.

        Both statements are restricted to `edge IS NULL`: this runs on EVERY open
        and only reconstructs rows that carry no explicit edge state. A row that
        already has `edge='replaces'` (or `'unknown'`) is authoritative and is
        never re-interpreted — in particular a free-text `note` containing a
        marker word can no longer downgrade a proven edge.
        """
        unproven = {r["claim_id"] for r in self.conn.execute(
            "SELECT DISTINCT claim_id FROM claim_events"
            " WHERE kind='created' AND detail LIKE '%unproven%'")}
        unproven |= {r["claim_id"] for r in self.conn.execute(
            "SELECT claim_id FROM claims WHERE note LIKE '%未证明%'")}
        # `edge IS NULL` is REQUIRED: only rows that never had an explicit edge
        # state are reconstructed. A row that already carries `edge='replaces'`
        # keeps it even if its free-text note happens to contain a marker word —
        # the durable field is authoritative and must never be overwritten by a
        # heuristic over the reason text.
        self.conn.execute(
            "UPDATE claims SET edge='unknown'"
            " WHERE replaces IS NOT NULL AND edge IS NULL AND claim_id IN"
            f" ({','.join('?' * len(unproven))})", tuple(unproven)) if unproven else None
        self.conn.execute(
            "UPDATE claims SET edge='replaces' WHERE replaces IS NOT NULL AND edge IS NULL")

    def close(self) -> None:
        self.conn.close()

    # ---- paths -----------------------------------------------------------
    def user_dir(self, uid: str) -> Path:
        return self.users_dir / dir_name(uid)

    def version_dir(self, uid: str, version: int) -> Path:
        return self.user_dir(uid) / "versions" / f"v{version}"

    def manifest_path(self, uid: str) -> Path:
        return self.user_dir(uid) / "manifest.json"

    def evidence_path(self, uid: str) -> Path:
        manifest = self.read_manifest(uid)
        return Path(manifest["evidence_path"]) if manifest else self.version_dir(uid, 0) / "evidence.jsonl"

    def diagnostics_path(self, uid: str) -> Path:
        return self.user_dir(uid) / "diagnostics.jsonl"

    def trace_path(self, uid: str, batch_id: str, attempt: int = 1) -> Path:
        """Attempt-scoped trace path: sanitised id + full-id digest + attempt number
        (no collision between 'x/y' and 'x?y', and retries never overwrite)."""
        digest = hashlib.sha256(batch_id.encode("utf-8")).hexdigest()[:12]
        safe = SAFE.sub("_", batch_id)[:32] or "batch"
        return self.user_dir(uid) / "trace" / f"{safe}-{digest}-a{attempt}.json"

    def _trace_attempts(self, uid: str, batch_id: str) -> int:
        digest = hashlib.sha256(batch_id.encode("utf-8")).hexdigest()[:12]
        safe = SAFE.sub("_", batch_id)[:32] or "batch"
        folder = self.user_dir(uid) / "trace"
        used: list[int] = []
        if folder.exists():
            for p in folder.glob(f"{safe}-{digest}-a*.json"):
                tail = p.stem.rsplit("-a", 1)[-1]
                if tail.isdigit():
                    used.append(int(tail))
        return (max(used) + 1) if used else 1

    def write_trace(self, uid: str, batch_id: str, record: dict) -> Path:
        path = self.trace_path(uid, batch_id, self._trace_attempts(uid, batch_id))
        atomic_write_text(path, json.dumps(record, ensure_ascii=False, indent=2))
        return path

    def read_manifest(self, uid: str) -> dict | None:
        path = self.manifest_path(uid)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def published_version(self, uid: str) -> int:
        manifest = self.read_manifest(uid)
        if manifest and isinstance(manifest.get("version"), int):
            return manifest["version"]
        return 0

    def summary_path(self, uid: str) -> Path:
        manifest = self.read_manifest(uid)
        return Path(manifest["summary_path"]) if manifest else self.version_dir(uid, 0) / "summary.md"

    def memory_path(self, uid: str) -> Path:
        manifest = self.read_manifest(uid)
        return Path(manifest["memory_path"]) if manifest else self.version_dir(uid, 0) / "memory.md"

    # ---- transactions ----------------------------------------------------
    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.conn.rollback()
            raise
        else:
            self.conn.commit()

    # ---- users / batches -------------------------------------------------
    def current_version(self, uid: str) -> int:
        row = self.conn.execute("SELECT version FROM users WHERE user_id=?", (uid,)).fetchone()
        return int(row["version"]) if row else 0

    def now_max(self, uid: str) -> str | None:
        row = self.conn.execute("SELECT now_max FROM users WHERE user_id=?", (uid,)).fetchone()
        return row["now_max"] if row else None

    def now_max_dt(self, uid: str):
        value = self.now_max(uid)
        return parse_iso(value) if value else None

    def ensure_user(self, uid: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM users WHERE user_id=?", (uid,)).fetchone()
        if row:
            return False
        self.conn.execute(
            "INSERT INTO users(user_id, created_at, now_max, version) VALUES(?,?,?,0)",
            (uid, _now_iso(), None),
        )
        self.conn.commit()
        self.publish(uid)
        return True

    def has_batch(self, uid: str, batch_id: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM batches WHERE user_id=? AND batch_id=?", (uid, batch_id)
        ).fetchone() is not None

    def batch_row(self, uid: str, batch_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM batches WHERE user_id=? AND batch_id=?", (uid, batch_id)
        ).fetchone()
        return dict(row) if row else None

    def record_batch(self, uid: str, batch_id: str, version: int, status: str,
                     note: str = "", fingerprint: str | None = None) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO batches(user_id,batch_id,applied_at,version,status,note,fingerprint)"
            " VALUES(?,?,?,?,?,?,?)",
            (uid, batch_id, _now_iso(), version, status, note, fingerprint),
        )

    def update_now(self, uid: str, now_iso: str) -> None:
        current = self.now_max_dt(uid)
        incoming = parse_iso(now_iso)
        if current is None or incoming > current:
            self.conn.execute("UPDATE users SET now_max=? WHERE user_id=?", (now_iso, uid))

    # ---- messages --------------------------------------------------------
    def get_message(self, uid: str, mid: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM messages WHERE user_id=? AND message_id=?", (uid, mid)
        ).fetchone()
        return dict(row) if row else None

    def add_message(self, uid: str, m: dict, batch_id: str) -> None:
        self.conn.execute(
            "INSERT INTO messages(user_id,message_id,conversation_id,role,message_at,text,"
            "ingested_at,batch_id) VALUES(?,?,?,?,?,?,?,?)",
            (uid, m["id"], m.get("conversation_id"), m["role"], m["message_at"], m["text"],
             _now_iso(), batch_id),
        )

    def messages(self, uid: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE user_id=? ORDER BY message_at, message_id", (uid,)
        ).fetchall()
        return [dict(r) for r in rows]

    def count_messages(self, uid: str) -> int:
        return int(self.conn.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE user_id=?", (uid,)
        ).fetchone()["n"])

    # ---- claims ----------------------------------------------------------
    def insert_claim(self, uid: str, claim: dict, sources: list[dict], version: int) -> str:
        cid = claim.get("claim_id") or new_claim_id()
        self.conn.execute(
            "INSERT INTO claims(claim_id,user_id,text,kind,status,importance,valid_from,valid_to,"
            "relative_phrase,resolved_from_phrase,event_at,subject,first_seen,last_updated,"
            "created_version,replaces,replaced_by,verification,note,edge)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (cid, uid, claim["text"], claim["kind"], claim["status"], claim.get("importance", "medium"),
             claim.get("valid_from"), claim.get("valid_to"), claim.get("relative_phrase"),
             claim.get("resolved_from_phrase"), claim.get("event_at"), claim.get("subject", "user"),
             _now_iso(), _now_iso(), version, claim.get("replaces"), None,
             claim.get("verification"), claim.get("note"), claim.get("edge")),
        )
        for s in sources:
            self.conn.execute(
                "INSERT OR IGNORE INTO claim_sources(claim_id,message_id,role,quote,message_at)"
                " VALUES(?,?,?,?,?)",
                (cid, s["message_id"], s["role"], s["quote"], s["message_at"]),
            )
        return cid

    def get_claim(self, uid: str, cid: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM claims WHERE user_id=? AND claim_id=?", (uid, cid)
        ).fetchone()
        return dict(row) if row else None

    def set_claim_status(self, uid: str, cid: str, status: str, *,
                         replaced_by: str | None = None, note: str | None = None,
                         verification: str | None = None) -> None:
        sets = ["status=?", "last_updated=?", "note=COALESCE(?,note)"]
        args: list[Any] = [status, _now_iso(), note]
        if replaced_by is not None:
            sets.append("replaced_by=?")
            args.append(replaced_by)
        if verification is not None:
            sets.append("verification=?")
            args.append(verification)
        args.extend([uid, cid])
        self.conn.execute(
            f"UPDATE claims SET {', '.join(sets)} WHERE user_id=? AND claim_id=?", args)

    def add_source(self, cid: str, s: dict) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO claim_sources(claim_id,message_id,role,quote,message_at)"
            " VALUES(?,?,?,?,?)",
            (cid, s["message_id"], s["role"], s["quote"], s["message_at"]),
        )

    def set_claim_bound(self, uid: str, cid: str, valid_to: str, note: str | None = None) -> None:
        """Record a derived (system) validity upper bound on an existing claim."""
        self.conn.execute(
            "UPDATE claims SET valid_to=?, last_updated=?, note=COALESCE(?,note)"
            " WHERE user_id=? AND claim_id=?",
            (valid_to, _now_iso(), note, uid, cid),
        )

    def add_event(self, uid: str, cid: str, kind: str, from_status: str | None,
                  to_status: str | None, detail: str = "") -> None:
        self.conn.execute(
            "INSERT INTO claim_events(claim_id,user_id,at,kind,from_status,to_status,detail)"
            " VALUES(?,?,?,?,?,?,?)",
            (cid, uid, _now_iso(), kind, from_status, to_status, detail),
        )

    def has_event(self, uid: str, cid: str, kind: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM claim_events WHERE user_id=? AND claim_id=? AND kind=?",
            (uid, cid, kind),
        ).fetchone() is not None

    def list_claims(self, uid: str, statuses: set[str] | None = None) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM claims WHERE user_id=? ORDER BY first_seen, claim_id", (uid,)
        ).fetchall()
        out = [dict(r) for r in rows]
        if statuses is not None:
            out = [c for c in out if c["status"] in statuses]
        return out

    def sources_for(self, claim_ids: list[str]) -> dict[str, list[dict]]:
        if not claim_ids:
            return {}
        marks = ",".join("?" for _ in claim_ids)
        rows = self.conn.execute(
            f"SELECT * FROM claim_sources WHERE claim_id IN ({marks}) ORDER BY claim_id, message_at",
            claim_ids,
        ).fetchall()
        out: dict[str, list[dict]] = {}
        for r in rows:
            out.setdefault(r["claim_id"], []).append(
                {"message_id": r["message_id"], "role": r["role"], "quote": r["quote"],
                 "message_at": r["message_at"]})
        return out

    def claim_source_messages(self, uid: str, cid: str) -> list[dict]:
        """Original evidence messages backing a claim (joined to our evidence store)."""
        rows = self.conn.execute(
            "SELECT s.message_id, s.role, s.message_at, s.quote, m.text"
            " FROM claim_sources s LEFT JOIN messages m"
            "   ON m.user_id=? AND m.message_id=s.message_id"
            " WHERE s.claim_id=? ORDER BY s.message_at",
            (uid, cid),
        ).fetchall()
        return [{"message_id": r["message_id"], "role": r["role"], "message_at": r["message_at"],
                 "quote": r["quote"], "text": r["text"]} for r in rows]

    # ---- versions & publication -----------------------------------------
    def bump_version(self, uid: str) -> int:
        v = self.current_version(uid) + 1
        self.conn.execute("UPDATE users SET version=? WHERE user_id=?", (v, uid))
        self.conn.execute(
            "INSERT OR REPLACE INTO versions(user_id,version,created_at) VALUES(?,?,?)",
            (uid, v, _now_iso()),
        )
        return v

    def finalize_version(self, uid: str, version: int, *, summary_text: str, memory_text: str,
                         evidence_count: int, claims_active: int, note: str = "",
                         rendered_now: str | None = None) -> None:
        self.conn.execute(
            "UPDATE versions SET summary_sha=?, memory_sha=?, evidence_count=?, claims_active=?,"
            " note=?, summary_text=?, memory_text=?, rendered_now=? WHERE user_id=? AND version=?",
            (sha256_text(summary_text), sha256_text(memory_text), evidence_count, claims_active,
             note, summary_text, memory_text, rendered_now, uid, version),
        )

    def rendered_now_text(self, uid: str) -> str | None:
        """The `now` the CURRENT version's material was actually rendered at.

        Used to decide whether a time-only batch must re-render the material and
        summary (so the current understanding stops being frozen at an old 此刻).

        Resolution order (never inferred from `now_max`, which may already have
        advanced without the material being regenerated):
          1. the version's recorded `rendered_now` (v2.2+ rows);
          2. otherwise the 「最近 now：」 line of the stored material itself — this
             covers v2.1 rows migrated in with a NULL rendered_now, so a legacy
             state still time-refreshes correctly;
          3. otherwise None (e.g. version 0 has no stored material: it is rebuilt
             fresh at publish, so there is no stale understanding to refresh).
        """
        version = self.current_version(uid)
        row = self.conn.execute(
            "SELECT rendered_now FROM versions WHERE user_id=? AND version=?",
            (uid, version),
        ).fetchone()
        if row and row["rendered_now"]:
            return row["rendered_now"]
        _, memory_text = self.version_texts(uid, version)
        return self._material_now(memory_text)

    _MATERIAL_NOW = re.compile(r"最近 now：\s*(\d\S*)")

    @classmethod
    def _material_now(cls, memory_text: str | None) -> str | None:
        if not memory_text:
            return None
        match = cls._MATERIAL_NOW.search(memory_text)
        return match.group(1).strip() if match else None

    def version_texts(self, uid: str, version: int) -> tuple[str | None, str | None]:
        row = self.conn.execute(
            "SELECT summary_text, memory_text FROM versions WHERE user_id=? AND version=?",
            (uid, version),
        ).fetchone()
        if not row:
            return None, None
        return row["summary_text"], row["memory_text"]

    def diagnose(self, uid: str, record: dict[str, Any]) -> None:
        with self.diagnostics_path(uid).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _render_memory(self, uid: str, version: int) -> str:
        from . import render
        claims = self.list_claims(uid)
        srcs = self.sources_for([c["claim_id"] for c in claims])
        msg_text = {m["message_id"]: m["text"] for m in self.messages(uid)}
        return render.render_memory(uid, version, self.now_max(uid) or "", claims, srcs, msg_text)

    def publish(self, uid: str) -> dict:
        """Publish the current DB version; manifest.json written LAST (atomic pointer)."""
        from . import render

        version = self.current_version(uid)
        summary_text, memory_text = self.version_texts(uid, version)
        if not summary_text:
            summary_text = render.empty_summary(uid)
        if not memory_text:
            memory_text = self._render_memory(uid, version)

        # Never overwrite an already published snapshot, including no-op batches.
        vdir = self.user_dir(uid) / "versions" / f"v{version}-{uuid.uuid4().hex}"
        atomic_write_text(vdir / "summary.md", summary_text)
        atomic_write_text(vdir / "memory.md", memory_text)

        msgs = self.messages(uid)
        evidence_text = render.render_evidence(uid, msgs)
        atomic_write_text(vdir / "evidence.jsonl", evidence_text)

        claims = self.list_claims(uid)
        manifest = {
            "user_id": uid,
            "version": version,
            "published_at": _now_iso(),
            "now_max": self.now_max(uid),
            "summary_path": str(vdir / "summary.md"),
            "memory_path": str(vdir / "memory.md"),
            "evidence_path": str(vdir / "evidence.jsonl"),
            "summary_sha": sha256_text(summary_text),
            "memory_sha": sha256_text(memory_text),
            "evidence_sha": sha256_text(evidence_text),
            "evidence_count": len(msgs),
            "claims_total": len(claims),
            "claims_active": sum(1 for c in claims if c["status"] in ("active", "planned", "temporary")),
        }
        atomic_write_text(self.manifest_path(uid), json.dumps(manifest, ensure_ascii=False, indent=2))
        return manifest

    def reconcile(self, uid: str) -> dict:
        """Republish the committed version if the pointer is stale/missing (recovery)."""
        manifest = self.read_manifest(uid)
        version = self.current_version(uid)
        if (manifest and manifest.get("version") == version
                and manifest.get("now_max") == self.now_max(uid)
                and manifest.get("evidence_count") == self.count_messages(uid)):
            try:
                if all(sha256_text(Path(manifest[f"{name}_path"]).read_text("utf-8"))
                       == manifest.get(f"{name}_sha") for name in ("summary", "memory", "evidence")):
                    return manifest
            except (OSError, KeyError):
                pass
        return self.publish(uid)

    def expiry_due(self, uid: str, now) -> list[dict]:
        due: list[dict] = []
        for c in self.list_claims(uid):
            if c["status"] not in ("active", "planned", "temporary", "unconfirmed"):
                continue
            end = parse_flexible(c.get("valid_to"))
            if end is None or end > now:
                continue
            # An unconfirmed claim's elapsed window is recorded ONCE and never
            # re-processed, so a weak claim neither raises confidence in time nor
            # causes per-batch churn.
            if c["status"] == "unconfirmed" and self.has_event(uid, c["claim_id"], "window_elapsed"):
                continue
            if c["kind"] == "undecided" and self.has_event(uid, c["claim_id"], "window_elapsed"):
                continue
            due.append(c)
        return due
