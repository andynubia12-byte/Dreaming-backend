"""8788 append-only events and independently retrieved stable memory.

Only memory-v1.sqlite is written. The injected source connection belongs to the
caller's isolated user workspace (the legacy messages schema has no user column).
Chat sources are additionally instance-scoped; optional user_id columns are
checked. A state directory is permanently bound to config.user_id / instance.
No model client or credential is created here. Successful operation retries return
the original result, even if a subsequent operation has advanced current().
"""
from __future__ import annotations

from contextlib import contextmanager
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3

from .config import Config
from .deepseek import DeepSeekClient
from .memory_agent import assistant_wire, _dispatch
from .memory_v1_prompts import EVENT_SYSTEM, STABLE_SYSTEM, STABLE_FINAL, TOOLS
from .providers import response_input
from .retrieval import TrialRetrieval, now_shanghai
from .summary_store import Published
from .util import approx_tokens


class MemoryV1Error(ValueError):
    """Invalid request, evidence, budget or incomplete model result."""


class MemoryV1Conflict(MemoryV1Error):
    """The expected version or operation identity no longer matches."""


_CITATION = re.compile(r"msg:[\w-]+#[\w-]+|chat#\d+")
_STABLE_HEADER = "# 稳定记忆层（stable.md，实验候选）"
_EVENT_HEADER = "# 完整事件摘要（原 idea.md，逐字保留）"


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _time(value):
    try:
        parsed = dt.datetime.fromisoformat(value)
        if parsed.utcoffset() is None:
            raise ValueError
        epoch = parsed.timestamp()
        if not math.isfinite(epoch):
            raise ValueError
        return epoch
    except (TypeError, ValueError, OverflowError):
        raise MemoryV1Error("now must be a timezone-aware ISO timestamp") from None


def _ids(value):
    if not isinstance(value, list) or any(not isinstance(s, str) or not s for s in value):
        raise MemoryV1Error("source_ids must be a list of nonempty canonical ids")
    return list(dict.fromkeys(value))


def _assemble(event, stable):
    return (f"{_STABLE_HEADER}\n\n{stable}\n\n{_EVENT_HEADER}\n\n{event}"
            if stable.strip() else event)


class MemoryV1:
    def __init__(self, state_dir: Path, config: Config):
        self.config = config
        self.path = Path(state_dir) / "memory-v1.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS owner (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    user_id TEXT NOT NULL, instance TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS versions (
                    number INTEGER PRIMARY KEY, memory_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS current (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    number INTEGER NOT NULL REFERENCES versions(number));
                CREATE TABLE IF NOT EXISTS operations (
                    operation_id TEXT PRIMARY KEY, input_hash TEXT NOT NULL,
                    result_json TEXT, covered_json TEXT NOT NULL DEFAULT '[]');
            """)
            db.execute("INSERT OR IGNORE INTO owner VALUES(1,?,?)",
                       (config.user_id, config.instance))
            self._owner(db)
            zero = dict(version=0, event_text="", stable_text="", text="", present=False,
                        published_at=None, now_max=None, sources=[], operation_id=None,
                        event_source_ids=[], stable_source_ids=[])
            db.execute("INSERT OR IGNORE INTO versions VALUES(0,?)", (_json(zero),))
            db.execute("INSERT OR IGNORE INTO current VALUES(1,0)")

    @contextmanager
    def _db(self):
        # Every connection is short lived; no database lock spans a model call.
        conn = sqlite3.connect(self.path, timeout=10)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _owner(self, db):
        if db.execute("SELECT user_id,instance FROM owner WHERE singleton=1").fetchone() != (
                self.config.user_id, self.config.instance):
            raise MemoryV1Error("memory state belongs to a different user or instance")

    @staticmethod
    def _current(db):
        return json.loads(db.execute("SELECT memory_json FROM versions JOIN current "
                                    "ON versions.number=current.number").fetchone()[0])

    def current(self) -> dict:
        with self._db() as db:
            self._owner(db)
            return self._current(db)

    def versions(self) -> list:
        """Published snapshots in increasing order; version 0 is also inspectable."""
        with self._db() as db:
            self._owner(db)
            return [json.loads(r[0]) for r in db.execute(
                "SELECT memory_json FROM versions ORDER BY number")]

    def version(self, number: int) -> dict:
        if type(number) is not int or number < 0:
            raise MemoryV1Error("version must be a nonnegative integer")
        with self._db() as db:
            self._owner(db)
            row = db.execute("SELECT memory_json FROM versions WHERE number=?", (number,)).fetchone()
        if row is None:
            raise MemoryV1Error("memory version not found")
        return json.loads(row[0])

    def published(self) -> Published:
        memory = self.current()
        return Published(present=memory["present"], version=memory["version"],
                         text=memory["text"], published_at=memory["published_at"],
                         now_max=memory["now_max"], user_id=self.config.user_id,
                         reason="" if memory["present"] else "no published v1 memory")

    @staticmethod
    def _retry(db, operation_id, fingerprint):
        row = db.execute("SELECT input_hash,result_json FROM operations WHERE operation_id=?",
                         (operation_id,)).fetchone()
        if row and row[0] != fingerprint:
            raise MemoryV1Conflict("operation_id reused with different inputs")
        return json.loads(row[1]) if row and row[1] else None

    def _begin(self, operation_id, fingerprint, expected_version):
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            self._owner(db)
            result = self._retry(db, operation_id, fingerprint)
            if result:
                return result, None, None
            old = self._current(db)
            if old["version"] != expected_version:
                raise MemoryV1Conflict("expected_version does not match current version")
            db.execute("INSERT OR IGNORE INTO operations(operation_id,input_hash) VALUES(?,?)",
                       (operation_id, fingerprint))
            covered = set()
            for row in db.execute("SELECT covered_json FROM operations WHERE result_json IS NOT NULL"):
                covered.update(json.loads(row[0]))
            return None, old, covered

    def _commit(self, operation_id, fingerprint, expected, candidate, covered):
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            self._owner(db)
            result = self._retry(db, operation_id, fingerprint)
            if result:
                return result
            old = self._current(db)
            if old["version"] != expected:
                raise MemoryV1Conflict("memory changed during generation; candidate not published")
            changed = (candidate["event_text"], candidate["stable_text"]) != (
                old["event_text"], old["stable_text"])
            memory = old
            if changed:
                memory = dict(candidate, version=expected + 1, present=True,
                              published_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                              operation_id=operation_id)
                memory["text"] = _assemble(memory["event_text"], memory["stable_text"])
                db.execute("INSERT INTO versions VALUES(?,?)", (memory["version"], _json(memory)))
                updated = db.execute("UPDATE current SET number=? WHERE singleton=1 AND number=?",
                                     (memory["version"], expected))
                if updated.rowcount != 1:
                    raise MemoryV1Conflict("memory changed during publication")
            result = dict(status="published" if changed else "noop", memory=memory,
                          version=memory["version"])
            db.execute("UPDATE operations SET result_json=?,covered_json=? WHERE operation_id=?",
                       (_json(result), _json(sorted(covered)), operation_id))
            return result

    def _retrieval(self, conn, now):
        r = TrialRetrieval(conn, self.config.instance, now)
        # Current schema: exports belong to the isolated workspace, chat_log has
        # instance ownership. Honour explicit per-row ownership if added later.
        columns = {table: {row['name'] for row in conn.execute(f"PRAGMA table_info({table})")}
                   for table in ("messages", "chat_log")}
        visible = []
        for doc in r.docs:
            table = "chat_log" if doc["kind"] == "chat" else "messages"
            if "user_id" in columns[table]:
                if table == "chat_log":
                    owner = conn.execute("SELECT user_id FROM chat_log WHERE id=?",
                                         (doc["chat_id"],)).fetchone()[0]
                else:
                    owner = conn.execute("SELECT user_id FROM messages WHERE conversation_id=? AND node_id=?",
                                         (doc["conv"], doc["node"])).fetchone()[0]
                if owner != self.config.user_id:
                    continue
            visible.append(doc)
        # Rebuild the existing retrieval corpus only when an owner filter applies.
        if len(visible) != len(r.docs):
            from .retrieval import _Corpus
            r.docs = visible
            r.by_sid = {d["sid"]: d for d in visible}
            r.corpus = _Corpus(visible)
        return r

    @staticmethod
    def _verify(ids, retrieval, now):
        docs = []
        for sid in ids:
            doc = retrieval.by_sid.get(sid)  # canonical ids only; no ambiguous prefix fallback
            if doc is None or doc["role"] != "user" or not doc["t"] < now:
                raise MemoryV1Error(f"invalid, non-user, unselected, foreign or future source: {sid}")
            docs.append(doc)
        return sorted(docs, key=lambda d: (d["t"], d["sid"]))

    def _call(self, client, messages, *, tools=None, response_json=False):
        budget = getattr(self.config, "max_input_tokens", 200_000)
        estimate = approx_tokens(_json({"input": response_input(messages), "tools": tools or []}))
        if estimate > budget:
            raise MemoryV1Error(f"full input exceeds token budget ({estimate}/{budget}); not truncated")
        headroom = max(16_000, getattr(self.config, "maint_max_output_tokens", 16_000))
        profile = getattr(client, "_provider", None)
        if profile is not None and profile.parameters.get("max_output_tokens", headroom) < 16_000:
            raise MemoryV1Error("summary profile requires at least 16000 output tokens")
        try:
            payload = client.chat(messages, max_tokens=headroom, temperature=0.0 if not tools else 0.2,
                                  response_json=response_json, tools=tools)
        except Exception as exc:
            raise MemoryV1Error("model request failed: " + type(exc).__name__) from None
        choices = payload.get("choices") or []
        if not choices or choices[0].get("finish_reason") not in ("stop", "tool_calls"):
            raise MemoryV1Error("model response did not finish completely")
        message = DeepSeekClient.message(payload)
        calls = message.get("tool_calls") or []
        if calls and (not tools or choices[0]["finish_reason"] != "tool_calls"):
            raise MemoryV1Error("unexpected model tool calls")
        if not calls and (choices[0]["finish_reason"] != "stop" or not message.get("content", "").strip()):
            raise MemoryV1Error("model response has no completed body")
        return message

    def _event(self, client, current, docs, now):
        evidence = [dict(id=d["sid"], role="user", message_at=now_shanghai(d["t"]),
                         text=d["text"], conversation_id=d["conv"]) for d in docs]
        user = (f"SIMULATED NOW: {now}. Relative time words refer to each item's OWN message_at.\n"
                f"CURRENT MEMORY DOCUMENT (full):\n{current}\n\nVERIFIED USER EVIDENCE:\n" +
                "\n".join(_json(item) for item in evidence))
        message = self._call(client, [{"role": "system", "content": EVENT_SYSTEM},
                                      {"role": "user", "content": user}], response_json=True)
        try:
            data = json.loads(message["content"])
        except (ValueError, TypeError):
            raise MemoryV1Error("event response must be complete valid JSON") from None
        if (not isinstance(data, dict) or not isinstance(data.get("section_markdown"), str)
                or not isinstance(data.get("summary"), str)
                or not isinstance(data.get("supersedes"), list)
                or any(not isinstance(s, str) for s in data["supersedes"])
                or type(data.get("confidence")) not in (int, float)
                or not 0 <= data["confidence"] <= 1):
            raise MemoryV1Error("invalid event response structure")
        sources = _ids(data.get("sources"))
        allowed = {d["sid"] for d in docs}
        if set(sources) - allowed:
            raise MemoryV1Error("event response cites sources outside verified batch")
        section = data["section_markdown"].strip()
        if not section:
            if sources:
                raise MemoryV1Error("empty event section cannot claim sources")
            return current, []
        expected_heading = "### 更新 " + dt.datetime.fromisoformat(now).date().isoformat()
        if section.splitlines()[0] != expected_heading:
            raise MemoryV1Error("event section must start with the batch update date")
        citations = set(_CITATION.findall(section))
        if not sources or not citations or citations - set(sources) or set(sources) - citations:
            raise MemoryV1Error("event citations must match verified response sources")
        return current.rstrip() + "\n\n" + section + "\n", sources

    def _stable(self, client, retrieval, now, emit):
        messages = [{"role": "system", "content": STABLE_SYSTEM}, {"role": "user", "content":
            f"SIMULATED NOW: {now}. 只有此刻之前的历史存在。\n"
            "请先自主检索，覆盖用户基本背景、对AI交流方式的明确要求/纠正、长期兴趣、重要关系现状。"}]
        seen = set()
        rounds = getattr(self.config, "memory_v1_tool_round_limit", 10)
        if type(rounds) is not int or rounds < 1:
            raise MemoryV1Error("memory_v1_tool_round_limit must be a positive integer")
        for turn in range(1, rounds + 1):
            emit("stable_retrieval", turn=turn)
            message = self._call(client, messages, tools=TOOLS)
            calls = message.get("tool_calls") or []
            messages.append(assistant_wire(message.get("content") or "", calls or None,
                                           message.get("reasoning_content"), message.get("_response_items")))
            if not calls:
                break
            for call in calls:
                fn = call.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except (ValueError, TypeError):
                    raise MemoryV1Error("invalid tool argument JSON") from None
                if not isinstance(args, dict) or not call.get("id"):
                    raise MemoryV1Error("invalid tool call structure")
                name = fn.get("name")
                if name not in {"search_history", "read_message", "expand"}:
                    raise MemoryV1Error("unknown stable retrieval tool")
                schema = next(t['function']['parameters'] for t in TOOLS
                              if t['function']['name'] == name)
                if set(args) - set(schema['properties']) or set(schema['required']) - set(args):
                    raise MemoryV1Error("invalid retrieval tool argument fields")
                for key, value in args.items():
                    field = schema['properties'][key]
                    expected = {'string': str, 'integer': int, 'array': list}[field['type']]
                    if type(value) is not expected or ('enum' in field and value not in field['enum']):
                        raise MemoryV1Error("invalid retrieval tool argument type")
                    if isinstance(value, list) and any(not isinstance(item, str) for item in value):
                        raise MemoryV1Error("retrieval keywords/entities must be strings")
                # Reject loose/foreign ids. Assistant rows remain labelled context.
                if name in {"read_message", "expand"}:
                    sid = args.get("source_id")
                    if sid not in retrieval.by_sid:
                        raise MemoryV1Error("tool requested an invalid or invisible source")
                emit("stable_tool_call", turn=turn, name=name, arguments=args)
                try:
                    result = _dispatch(name, args, retrieval)
                except (ValueError, TypeError):
                    raise MemoryV1Error("invalid retrieval tool arguments") from None
                items = result.get("results", []) + result.get("items", [])
                if result.get("item"):
                    items.append(result["item"])
                # Read/expand provide full evidence: budget overflow fails later,
                # rather than silently trimming the originals to MAX_TEXT.
                for item in items:
                    if "text" in item:
                        item["text"] = retrieval.by_sid[item["source_id"]]["text"]
                        item["truncated"] = False
                seen.update(item["source_id"] for item in items if item.get("role") == "user")
                emit("stable_tool_result", turn=turn, name=name, result=result)
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": _json(result)})
        messages.append({"role": "user", "content": STABLE_FINAL})
        emit("stable_synthesis")
        markdown = self._call(client, messages)["content"].strip() + "\n"
        section, found_sections, sources = None, [], set()
        for line in markdown.splitlines():
            if not line.strip():
                continue
            match = re.fullmatch(r"## ([ABCD])\. .+", line)
            if match:
                section = match[1]
                found_sections.append(section)
                continue
            if section is None or line.lstrip().startswith("#"):
                raise MemoryV1Error("invalid stable section structure")
            cited = set(_CITATION.findall(line))
            if cited - seen:
                raise MemoryV1Error("stable source was not returned this round as role=user")
            if section != "D" and not cited:
                raise MemoryV1Error("each stable conclusion requires a retrieved user source")
            if section == "C" and (not re.search(r"\d{4}-\d{2}-\d{2}", line)
                                   or not any(word in line for word in ("当时", "现在", "不确定"))):
                raise MemoryV1Error("stable state requires a date and temporal status")
            sources.update(cited)
        if found_sections != list("ABCD"):
            raise MemoryV1Error("stable response requires ordered A/B/C/D sections")
        return markdown, sorted(sources)

    def execute(self, operation: str, *, conn: sqlite3.Connection, client, now: str,
                operation_id: str, expected_version: int, batches: list | None = None,
                source_ids: list | None = None, emit=None) -> dict:
        if operation not in {"initialize", "append", "refresh_stable"}:
            raise MemoryV1Error("unknown memory operation")
        if not isinstance(operation_id, str) or not operation_id.strip():
            raise MemoryV1Error("operation_id is required")
        if type(expected_version) is not int or expected_version < 0:
            raise MemoryV1Error("expected_version must be a nonnegative integer")
        epoch = _time(now)
        if operation == "initialize":
            if source_ids is not None or not isinstance(batches, list) or not batches:
                raise MemoryV1Error("initialize requires nonempty batches and no source_ids")
            prepared = []
            for batch in batches:
                if not isinstance(batch, dict) or set(batch) != {"now", "source_ids"}:
                    raise MemoryV1Error("each batch requires now and source_ids")
                when = _time(batch["now"])
                ids = _ids(batch["source_ids"])
                if when > epoch or not ids:
                    raise MemoryV1Error("batch must have evidence and must not be after operation now")
                prepared.append((when, batch["now"], ids))
            prepared.sort(key=lambda b: b[0])
        elif operation == "append":
            if batches is not None or not source_ids:
                raise MemoryV1Error("append requires explicit nonempty source_ids and no batches")
            prepared = [(epoch, now, _ids(source_ids))]
        else:
            if batches is not None or source_ids is not None:
                raise MemoryV1Error("refresh_stable uses autonomous retrieval, no explicit sources/batches")
            prepared = []
        fingerprint = hashlib.sha256(_json(dict(operation=operation, now=now,
            expected_version=expected_version, batches=batches, source_ids=source_ids)).encode()).hexdigest()
        result, old, covered = self._begin(operation_id, fingerprint, expected_version)
        if result:
            return result
        if operation == "initialize" and old["version"] != 0:
            raise MemoryV1Conflict("initialize requires empty version 0")
        if old["now_max"] is not None and epoch < _time(old["now_max"]):
            raise MemoryV1Error("operation now cannot precede the published now_max")
        # Materialise and validate ALL requested evidence before the first call.
        previous_factory = conn.row_factory
        try:
            conn.row_factory = sqlite3.Row
            retrieval = self._retrieval(conn, epoch)
        finally:
            conn.row_factory = previous_factory
        verified = [(batch_now, self._verify(ids, retrieval, when)) for when, batch_now, ids in prepared]
        candidate = dict(old, now_max=now)
        notify = emit or (lambda phase, **data: None)
        new_covered = set()
        if operation == "refresh_stable":
            candidate["stable_text"], candidate["stable_source_ids"] = self._stable(
                client, retrieval, now, notify)
        else:
            event = old["event_text"] or "# 长期记忆\n"
            event_sources = set(old["event_source_ids"])
            for index, (batch_now, docs) in enumerate(verified):
                docs = [d for d in docs if d["sid"] not in covered | new_covered]
                if not docs:
                    continue
                notify("event_generation", batch=index + 1, now=batch_now,
                       source_ids=[d["sid"] for d in docs])
                event, used = self._event(client, event, docs, batch_now)
                event_sources.update(used)
                new_covered.update(d["sid"] for d in docs)
            # An empty section must not publish the seed as user memory.
            if event != "# 长期记忆\n":
                candidate["event_text"] = event
            candidate["event_source_ids"] = sorted(event_sources)
        candidate["sources"] = sorted(set(candidate["event_source_ids"]) | set(candidate["stable_source_ids"]))
        notify("publication_started", expected_version=expected_version)
        result = self._commit(operation_id, fingerprint, expected_version, candidate, new_covered)
        # Publication is already durable; an observer cannot turn it into a failed operation.
        try:
            notify("publication_complete", status=result["status"], version=result["version"])
        except Exception:
            pass
        return result
