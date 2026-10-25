"""Batch maintenance pipeline.

Per batch, in order:
  1. strict tz-aware validation, future-evidence rejection, and now-regression
     rejection (no writes)
  2. idempotency: identical batch_id -> duplicate; same id with changed content
     -> explicit conflict; duplicate message_id -> skip; conflicting message_id
     aborts the whole batch
  3. propose claims (model) -> deterministic checks (source, verbatim quote,
     role, time, derivable dates, legal/ordered ranges) -> verify (model) with
     the referenced target claim + its original evidence
  4. apply accepted ops + deterministic expiry inside ONE transaction, storing
     the FULL version text (summary + memory) in the database
  5. publish the committed version atomically (manifest pointer written last);
     a publish failure leaves the previous complete version visible and is
     recoverable on the next call.

The summary is generated ONLY from the memory material (never from an old
summary). It is free model text; see the report for the residual limitation.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from typing import Any

from . import render
from .model import Model, ModelError
from .prompts import prompt_hashes
from .store import Store
from .timeutil import (
    TimeError,
    dates_in_text,
    parse_flexible,
    parse_iso,
    resolve_bound_end,
    resolve_relative_phrase,
    weekday_dates_in_text,
)

USER_FACT_KINDS = {"user_fact", "preference", "relation", "goal"}
KNOWN_OPS = {"add", "correct", "resolve", "retract", "confirm"}
KNOWN_KINDS = {"user_fact", "preference", "relation", "event", "plan", "temporary_state", "goal",
               "undecided"}
PLAN_KINDS = {"plan", "temporary_state"}
# Kinds whose applicability window is ended by a determinable period boundary
# (e.g. '月底' -> last day of the source month). An unresolved/undecided item is
# never turned into a completed fact; it only becomes "window elapsed".
BOUND_KINDS = {"plan", "temporary_state", "undecided"}
RESOLUTIONS = {"completed", "abandoned", "in_progress"}
IN_FORCE = ("active", "planned", "temporary", "in_progress")

# --- v2.4 candidate controls (all bounded to exactly one attempt) ----------
# Ops that assert a REPLACEMENT relation: the new evidence must independently
# show the target claim no longer holds. Without that proof the engine keeps the
# target claim and records the new content as its own claim instead of
# superseding/retracting on a merely compatible clarification.
RELATION_OPS = {"correct", "retract"}
# A minimal, explicit relation enum for replacement ops. The verifier must state
# HOW the new statement relates to the target claim:
#   "replaces"   -> the new evidence independently shows the target no longer
#                   holds (a real correction/retraction);
#   "compatible" -> the new statement can hold together with the target (a
#                   clarification or a denial of someone ELSE's reading); the
#                   target is kept and the new content is recorded on its own;
#   "unknown"    -> cannot tell. Kept as a PENDING relation with uncertainty
#                   (target -> contested, new content -> unconfirmed), never as
#                   two independent confirmed facts.
# Missing / malformed / non-string values all map to "unknown" (fail-safe).
RELATION_REPLACES = "replaces"
RELATION_COMPATIBLE = "compatible"
RELATION_UNKNOWN = "unknown"
RELATION_VALUES = {RELATION_REPLACES, RELATION_COMPATIBLE, RELATION_UNKNOWN}
# At most one bounded re-proposal round after the first propose/verify round.
MAX_REPAIR_ROUNDS = 1
# At most one constrained summary regeneration when the summary cites ids that
# do not resolve back to a known message/claim.
MAX_SUMMARY_REGEN = 1

# Citation syntax: a citation is a bracketed token written with only id
# characters (ASCII letters/digits/._:-#). Whether a token counts as a citation
# does NOT depend on it containing a digit. A bracketed token that appears
# VERBATIM inside the provided material or any source message is treated as
# quoted source text (e.g. a business model/part number like [AB12] or a
# bracketed list), not as a citation -- a general "quoted from input" rule, not
# a word/model whitelist.
#
# The bracket body is scanned with NO length cap: real ids can be long (e.g. a
# 77-character `msg:<uuid>#<uuid>` or `chat#123`) and a combined bracket can hold
# several of them. A length cap would silently skip such a token and let an
# unknown id publish unchecked, so every bracket body is examined instead.
_BRACKETED = re.compile(r"\[([^\[\]]*)\]")
# ASCII id characters plus `#`, which occurs in real `chat#123` / `msg:…#…` ids.
_CITATION_SHAPE = re.compile(r"^[A-Za-z0-9_.:\-#]+$")
# Separators a model may use to pack SEVERAL ids into one bracket, e.g.
# `[id-a; id-b]` or `[id-a, missing-source]`. The combined form is forbidden by
# contract, but its parts are still parsed so every id inside is validated.
_CITATION_SEPARATORS = re.compile(r"[;,\uFF0C\uFF1B\u3001]")

# --- v2.5 time-metadata guards --------------------------------------------
# Clock evidence in the SOURCE text: an explicit time-of-day the user actually
# stated. Day-part words alone (早上/晚上/中午…) are NOT clock evidence — they do
# not pin an instant. Used only to decide whether a model-supplied `event_at`
# that merely repeats the message instant is backed by the source.
_CLOCK_EVIDENCE = re.compile(
    r"\d{1,2}\s*[:：]\s*\d{2}"      # 21:20 / 21：20
    r"|\d{1,2}\s*点"                # 21点 / 9点
    r"|\d{1,2}\s*时"                # 21时
    r"|[一二三四五六七八九十]{1,3}\s*点"   # 九点 / 十一点
    r"|[一二三四五六七八九十]{1,3}\s*时"   # 九时
    r"|零点|午夜|正午|子夜|整点"
)
# Claim statuses whose window may still be open (i.e. eligible for expiry).
OPEN_STATUSES = ("active", "planned", "temporary")


def _norm_text(value: str | None) -> str:
    return re.sub(r"\s+", "", value or "")


def _is_verbatim(quote: str | None, text: str | None) -> bool:
    q, t = _norm_text(quote), _norm_text(text)
    return len(q) >= 2 and q in t


def _fingerprint(prepared: list[dict], now: dt.datetime) -> str:
    items = [json.dumps(
        {"id": m["id"], "role": m["role"], "message_at": m["message_at"],
         "conversation_id": m.get("conversation_id"), "text": m["text"]},
        ensure_ascii=False, sort_keys=True) for m in prepared]
    items.append(now.astimezone(dt.timezone.utc).isoformat())
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()


class Engine:
    def __init__(self, store: Store, model: Model, prompts: dict[str, str]) -> None:
        self.store = store
        self.model = model
        self.prompts = prompts
        self.hashes = prompt_hashes(prompts)

    # ---- result helpers --------------------------------------------------
    def _paths(self, uid: str) -> dict[str, str]:
        return {
            "summary_path": str(self.store.summary_path(uid)),
            "memory_path": str(self.store.memory_path(uid)),
            "evidence_path": str(self.store.evidence_path(uid)),
        }

    def _result(self, uid: str, status: str, version: int, **extra: Any) -> dict[str, Any]:
        out: dict[str, Any] = {
            "status": status,
            "user_id": uid,
            "version": version,
            "published_version": self.store.published_version(uid),
            "prompt_hashes": self.hashes,
            **self._paths(uid),
        }
        out.update({k: v for k, v in extra.items() if v is not None})
        return out

    # ---- main entry ------------------------------------------------------
    def run_batch(self, batch: dict) -> dict[str, Any]:
        uid = (batch or {}).get("user_id")
        bid = (batch or {}).get("batch_id")
        raw_now = (batch or {}).get("now")
        raw_msgs = (batch or {}).get("messages") or []
        if not isinstance(uid, str) or not uid or not isinstance(bid, str) or not bid:
            return {"status": "error", "user_id": uid, "batch_id": bid, "version": None,
                    "error": "user_id and batch_id are required", "prompt_hashes": self.hashes}

        trace: dict[str, Any] = {"user_id": uid, "batch_id": bid, "now": raw_now,
                                 "new_messages": [m.get("id") for m in raw_msgs if isinstance(m, dict)]}
        # recovery: republish any committed-but-unpublished version before new work
        try:
            self.store.ensure_user(uid)
            self.store.reconcile(uid)
        except BaseException as exc:  # noqa: BLE001
            return self._early(uid, trace, "error", error=f"reconcile: {type(exc).__name__}: {exc}")

        # --- 1. strict time validation ---
        try:
            now = parse_iso(raw_now)
        except TimeError as exc:
            return self._early(uid, trace, "rejected_time", error=str(exc))
        prepared: list[dict] = []
        seen: set[str] = set()
        for m in raw_msgs:
            if not isinstance(m, dict):
                return self._early(uid, trace, "error", error="message must be an object")
            for key in ("id", "role", "message_at", "text"):
                if not isinstance(m.get(key), str) or not m.get(key):
                    return self._early(uid, trace, "error",
                                       error=f"message missing/invalid field: {key}")
            if m["role"] not in ("user", "assistant"):
                return self._early(uid, trace, "error", error="role must be user|assistant")
            try:
                at = parse_iso(m["message_at"])
            except TimeError as exc:
                return self._early(uid, trace, "rejected_time", error=str(exc), rejected_message=m["id"])
            if at > now:
                return self._early(uid, trace, "rejected_future", rejected_message=m["id"],
                                   error=f"message {m['id']} at {m['message_at']} is after now {raw_now}")
            if m["id"] in seen:
                return self._early(uid, trace, "conflict", rejected_message=m["id"],
                                   error=f"duplicate message id within batch: {m['id']}")
            seen.add(m["id"])
            prepared.append({"id": m["id"], "role": m["role"], "message_at": at.isoformat(),
                             "conversation_id": m.get("conversation_id"), "text": m["text"]})

        # --- 2. idempotency & conflict ---
        fingerprint = _fingerprint(prepared, now)
        existing_batch = self.store.batch_row(uid, bid)
        if existing_batch:
            if existing_batch.get("fingerprint") == fingerprint:
                return self._early(uid, trace, "duplicate",
                                   note=f"batch {bid} already applied (identical content)")
            return self._early(uid, trace, "conflict",
                               error=f"batch_id {bid} already applied with different content")
        previous = self.store.now_max_dt(uid)
        if previous is not None and now < previous:
            return self._early(uid, trace, "rejected_time",
                               error="now is earlier than the last accepted now")
        # The `now` the CURRENT version's material was last rendered at. A batch
        # that advances past it must re-render material + summary, otherwise a
        # time-only batch freezes the current understanding at a stale 此刻.
        rendered_now_text = self.store.rendered_now_text(uid)
        try:
            prev_rendered_now = parse_iso(rendered_now_text) if rendered_now_text else None
        except TimeError:
            prev_rendered_now = None
        new_msgs: list[dict] = []
        dup_ids: list[str] = []
        for m in prepared:
            existing = self.store.get_message(uid, m["id"])
            if existing:
                same = (existing["role"] == m["role"]
                        and (existing["conversation_id"] or None) == (m["conversation_id"] or None)
                        and existing["message_at"] == m["message_at"]
                        and existing["text"] == m["text"])
                if same:
                    dup_ids.append(m["id"])
                    continue
                return self._early(uid, trace, "conflict", rejected_message=m["id"],
                                   error=f"message id {m['id']} conflicts with stored content")
            new_msgs.append(m)

        new_by_id = {m["id"]: m for m in new_msgs}
        dropped: list[dict] = []
        accepted: list[tuple[dict, dict]] = []
        model_calls = {"propose": 0, "verify": 0, "summarize": 0}

        # --- 3. propose + deterministic checks + verify ---
        if new_msgs:
            payload = {
                "now": now.isoformat(),
                "existing_claims": [{**self._claim_view(c),
                                     "evidence": self.store.claim_source_messages(uid, c["claim_id"])}
                                    for c in self.store.list_claims(uid)],
                "new_messages": [{"id": m["id"], "role": m["role"], "message_at": m["message_at"],
                                  "conversation_id": m["conversation_id"], "text": m["text"]}
                                 for m in new_msgs],
            }
            trace["propose_input"] = payload
            try:
                prop = self.model.propose(self.prompts["propose"], payload)
                model_calls["propose"] += 1
                trace["propose_response"] = prop
                if not isinstance(prop, dict):
                    raise ModelError("propose result is not a JSON object")
                candidates = self._check_ops(uid, prop.get("operations"), new_msgs, dropped)
                trace["candidates"] = candidates
                if candidates:
                    vpayload = {"now": now.isoformat(),
                                "claims": [self._verify_item(uid, i, c, new_by_id)
                                           for i, c in enumerate(candidates)]}
                    trace["verify_input"] = vpayload
                    vres = self.model.verify(self.prompts["verify"], vpayload)
                    model_calls["verify"] += 1
                    trace["verify_response"] = vres
                    accepted = self._process_verdicts(candidates, vres, dropped)
                # --- 3b. ONE bounded re-proposal round (repair) ---
                # If a proposed item was dropped, or accepted only as unconfirmed
                # (partial), give the model exactly one chance to re-propose a
                # minimal fix from the SAME original evidence plus the specific
                # verify feedback. The repaired candidate must pass the full
                # deterministic checks and an INDEPENDENT verify; `corrected_text`
                # is never adopted directly and partial is never auto-promoted.
                repaired = self._repair_round(uid, now, payload, new_msgs, new_by_id,
                                              accepted, dropped, model_calls, trace)
                accepted = repaired["accepted"]
                trace["coverage_notes"] = self._coverage_notes(accepted, new_msgs)
                trace["accepted"] = [
                    {"op": c["op"], "text": c["text"], "kind": c["kind"],
                     "verdict": v.get("verdict"), "issue": v.get("issue"),
                     "unconfirmed": bool(v.get("_unconfirmed")), "guards": c.get("guards", [])}
                    for c, v in accepted]
            except BaseException as exc:  # noqa: BLE001
                trace["error"] = f"{type(exc).__name__}: {exc}"
                trace["dropped"] = dropped
                tp, terr = self._safe_trace(uid, bid, trace)
                return self._fail(uid, bid, f"propose/verify: {type(exc).__name__}: {exc}",
                                  model_calls, trace_path=tp, trace_error=terr)

        trace["state_before"] = self._state_snapshot(uid)
        # --- 4. apply in one transaction (+ summarize inside it) ---
        applied: list[dict] = []
        expired: list[dict] = []
        pending: dict | None = None
        version = self.store.current_version(uid)
        try:
            with self.store.transaction():
                self.store.update_now(uid, now.isoformat())
                for m in new_msgs:
                    self.store.add_message(uid, m, bid)
                for c, v in accepted:
                    cid = self._apply_op(uid, c, v)
                    if cid:
                        applied.append({"op": c["op"], "claim_id": cid, "text": c["text"]})
                # v2.5: a plan/undecided/temporary claim whose SOURCE states a
                # determinable calendar range but which has no valid_to yet must
                # have that range upper bound recorded, so a pure `now` advance
                # can expire it in the material/DB — not only in the summary.
                # Runs inside this transaction (published with version/material/
                # summary; a later failure rolls it back too).
                bounds_backfilled = self._backfill_bounds(uid, trace)
                for c in self.store.expiry_due(uid, now):
                    self._expire(uid, c)
                    expired.append({"claim_id": c["claim_id"], "text": c["text"], "was": c["status"]})
                # Time advancing past what the current material reflects is itself
                # a durable change: the material and summary must be rebuilt at the
                # real now (plan/temporary windows still expire only on their own
                # evidence, never promoted to "completed").
                time_advanced = prev_rendered_now is not None and now > prev_rendered_now
                time_refresh = bool(time_advanced and not (applied or expired or bounds_backfilled))
                trace["time_advanced"] = time_advanced
                changed = bool(applied or expired or bounds_backfilled or time_advanced)
                if changed:
                    version = self.store.bump_version(uid)
                    material = self.store._render_memory(uid, version)
                    trace["summary_input"] = material
                    try:
                        summary_md = self.model.summarize(
                            self.prompts["summarize"],
                            {"now": now.isoformat(), "memory_material": material})
                        model_calls["summarize"] += 1
                    except BaseException as exc:  # noqa: BLE001
                        raise ModelError(f"summarize: {type(exc).__name__}: {exc}") from exc
                    if not (summary_md or "").strip():
                        raise ModelError("summarize returned empty content")
                    trace["summary_response"] = summary_md
                    # v2.4: a summary may only cite real message/claim ids. One
                    # constrained regeneration is allowed; otherwise fail closed.
                    summary_md = self._check_citations(uid, now, material, summary_md,
                                                       model_calls, trace)
                    pending = {"material": material, "summary": summary_md}
                    self.store.finalize_version(
                        uid, version, summary_text=summary_md, memory_text=material,
                        evidence_count=self.store.count_messages(uid),
                        claims_active=sum(1 for c in self.store.list_claims(uid) if c["status"] in IN_FORCE),
                        note=(f"applied={len(applied)} expired={len(expired)}"
                              + (" time_refresh" if time_refresh else "")),
                        rendered_now=now.isoformat())
                else:
                    version = self.store.current_version(uid)
                self.store.record_batch(uid, bid, version, "ok" if changed else "noop",
                                        note=f"new={len(new_msgs)} dup={len(dup_ids)} "
                                             f"applied={len(applied)} expired={len(expired)}",
                                        fingerprint=fingerprint)
        except BaseException as exc:  # noqa: BLE001
            trace["error"] = f"apply: {type(exc).__name__}: {exc}"
            trace["dropped"] = dropped
            trace["state_after_failed"] = self._state_snapshot(uid)  # transaction rolled back
            tp, terr = self._safe_trace(uid, bid, trace)
            return self._fail(uid, bid, f"apply: {type(exc).__name__}: {exc}", model_calls,
                              trace_path=tp, trace_error=terr)

        # --- 5. publish the committed version (manifest pointer last) ---
        try:
            manifest = self.store.publish(uid)
        except BaseException as exc:  # noqa: BLE001
            for entry in dropped:
                self.store.diagnose(uid, {"type": "dropped_op", "batch_id": bid, **entry})
            trace.update({"applied": applied, "expired": expired, "version": version,
                          "dropped": dropped, "publish_error": f"{type(exc).__name__}: {exc}",
                          "state_after": self._state_snapshot(uid)})
            tp, terr = self._safe_trace(uid, bid, trace)
            return self._result(uid, "committed_unpublished", version, batch_id=bid,
                                committed=True, recoverable=True,
                                error=f"publish failed (recoverable): {type(exc).__name__}: {exc}",
                                claims_applied=len(applied), expired=expired, model_calls=model_calls,
                                prompt_hashes=self.hashes, trace_path=tp, trace_error=terr,
                                state_after=trace["state_after"])
        for entry in dropped:
            self.store.diagnose(uid, {"type": "dropped_op", "batch_id": bid, **entry})
        status = "ok" if pending else "noop"
        trace.update({"applied": applied, "expired": expired, "version": version,
                      "dropped": dropped, "summary_regenerated": bool(pending),
                      "time_refresh": time_refresh,
                      "state_after": self._state_snapshot(uid),
                      "manifest_version": manifest.get("version"),
                      "published_version": self.store.published_version(uid)})
        trace_path, trace_error = self._safe_trace(uid, bid, trace)
        if pending:
            note = (f"time-advance refresh at {now.isoformat()}" if time_refresh
                    else f"applied={len(applied)} expired={len(expired)}")
        else:
            note = ("no durable change; evidence recorded" if new_msgs
                    else "time-only batch, now not advanced")
        return self._result(
            uid, status, version, batch_id=bid,
            new_messages=len(new_msgs), duplicate_messages=len(dup_ids),
            claims_applied=len(applied), applied=applied, expired=expired,
            dropped_ops=dropped, model_calls=model_calls,
            summary_regenerated=bool(pending), time_refresh=time_refresh, manifest=manifest,
            trace_path=trace_path, trace_error=trace_error,
            bounds_backfilled=list(trace.get("bounds_backfilled") or []),
            repair_triggered=bool(trace.get("repair", {}).get("triggered")),
            repair_applied=len(trace.get("repair", {}).get("accepted") or []),
            summary_citations_retried=bool(
                trace.get("summary_citations", {}).get("regenerated")),
            note=note)

    # ---- helpers ---------------------------------------------------------
    def _fail(self, uid: str, bid: str, error: str, model_calls: dict, **extra: Any) -> dict[str, Any]:
        return self._result(uid, "error", self.store.current_version(uid), batch_id=bid,
                            error=error, model_calls=model_calls, **extra)

    def _early(self, uid: str, trace: dict, status: str, **kw: Any) -> dict[str, Any]:
        """Record a terminal state for early exits (reject/duplicate/conflict) too."""
        trace["early_exit"] = status
        if "error" in kw:
            trace["error"] = kw["error"]
        tp, terr = self._safe_trace(uid, trace["batch_id"], trace)
        return self._result(uid, status, self.store.current_version(uid),
                            batch_id=trace["batch_id"], trace_path=tp, trace_error=terr, **kw)

    def _safe_trace(self, uid: str, bid: str, trace: dict) -> tuple[str | None, str | None]:
        """Write the per-stage attribution trace (attempt-scoped, never overwritten).

        Tracing never breaks a batch; a failure is returned so the caller can
        surface the observability gap instead of silently claiming a full trace.
        """
        try:
            return str(self.store.write_trace(uid, bid, trace)), None
        except BaseException as exc:  # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}"

    def _state_snapshot(self, uid: str) -> list[dict]:
        return [{"id": c["claim_id"], "status": c["status"], "kind": c["kind"],
                 "replaces": c["replaces"], "replaced_by": c["replaced_by"],
                 "edge": c.get("edge")}
                for c in self.store.list_claims(uid)]

    def _coverage_notes(self, accepted: list[tuple[dict, dict]], new_msgs: list[dict]) -> list[dict]:
        """DIAGNOSTIC only (never fails a batch).

        When a confirm is accepted, flag new user messages whose content is not
        covered by an accepted add/correct and whose confirm quote does not match
        the whole message. Exception coverage is thus observable, not enforced;
        selection quality remains pending verification.
        """
        if not any(c["op"] == "confirm" for c, _ in accepted):
            return []
        covered = {mid for c, _ in accepted if c["op"] in ("add", "correct") for mid in c["source_ids"]}
        notes: list[dict] = []
        for m in new_msgs:
            if m["role"] != "user" or m["id"] in covered:
                continue
            for c, _ in accepted:
                if c["op"] != "confirm" or m["id"] not in c["source_ids"]:
                    continue
                quotes = [s["quote"] for s in c["sources"] if s["message_id"] == m["id"]]
                if not any(_norm_text(q) == _norm_text(m["text"]) for q in quotes):
                    notes.append({"message_id": m["id"], "note":
                                  "confirm quote covers only part of the message; extra "
                                  "content/exception was not saved (selection quality pending)"})
                    break
        return notes

    def _claim_view(self, c: dict) -> dict:
        return {"id": c["claim_id"], "text": c["text"], "kind": c["kind"], "status": c["status"],
                "valid_from": c["valid_from"], "valid_to": c["valid_to"],
                "relative_phrase": c["relative_phrase"], "event_at": c["event_at"],
                "verification": c["verification"], "note": c["note"],
                "replaces": c["replaces"], "edge": c.get("edge")}

    def _target_view(self, uid: str, cid: str) -> dict | None:
        c = self.store.get_claim(uid, cid)
        if not c:
            return None
        return {"id": c["claim_id"], "text": c["text"], "kind": c["kind"], "status": c["status"],
                "valid_from": c["valid_from"], "valid_to": c["valid_to"],
                "relative_phrase": c["relative_phrase"], "event_at": c["event_at"],
                "verification": c["verification"], "note": c["note"],
                "replaces": c["replaces"], "edge": c.get("edge")}

    def _check_ops(self, uid: str, ops: list, new_msgs: list[dict], dropped: list[dict]) -> list[dict]:
        # Protocol structure must fail closed (no commit, retryable); only fully
        # formed elements proceed to the semantic checks below.
        if ops is None:
            raise ModelError("propose result missing 'operations' array")
        if not isinstance(ops, list):
            raise ModelError("propose result 'operations' is not an array")
        by_id = {m["id"]: m for m in new_msgs}
        out: list[dict] = []
        for op in ops:
            if not isinstance(op, dict):
                raise ModelError("operation element is not an object")
            if not isinstance(op.get("op"), str):
                raise ModelError("operation element missing string 'op'")
            if not isinstance(op.get("text"), str):
                raise ModelError("operation element missing string 'text'")
            name = op["op"].strip()
            text = op["text"].strip()
            kind = (op.get("kind") or "user_fact").strip()
            if name not in KNOWN_OPS:
                dropped.append({"op": name, "text": text, "reason": "unknown op"})
                continue
            if kind not in KNOWN_KINDS:
                kind = "user_fact"
            if not text:
                dropped.append({"op": name, "reason": "empty text"})
                continue
            resolution = (op.get("resolution") or "").strip() if isinstance(op.get("resolution"), str) else ""
            if name == "resolve" and resolution not in RESOLUTIONS:
                dropped.append({"op": name, "text": text,
                                "reason": "resolve requires explicit resolution completed|abandoned|in_progress"})
                continue
            ev_ids = [x for x in (op.get("evidence_message_ids") or []) if isinstance(x, str) and x in by_id]
            if not ev_ids:
                dropped.append({"op": name, "text": text, "reason": "no new-message evidence"})
                continue
            srcs: list[dict] = []
            bad: str | None = None
            for q in (op.get("quotes") or []):
                if not isinstance(q, dict):
                    continue
                mid = q.get("message_id")
                quote = q.get("quote") or ""
                if mid not in by_id:
                    bad = f"quote cites non-batch message {mid}"
                    break
                if not _is_verbatim(quote, by_id[mid]["text"]):
                    bad = f"quote not verbatim in {mid}"
                    break
                if mid not in {s["message_id"] for s in srcs}:
                    srcs.append({"message_id": mid, "role": by_id[mid]["role"],
                                 "quote": quote, "message_at": by_id[mid]["message_at"]})
            if bad or not srcs:
                dropped.append({"op": name, "text": text, "reason": bad or "no verbatim quotes"})
                continue
            roles = {s["role"] for s in srcs}
            if kind in USER_FACT_KINDS and roles == {"assistant"}:
                dropped.append({"op": name, "text": text,
                                "reason": "user fact supported only by assistant messages"})
                continue
            target = op.get("target_claim_id")
            if name in ("correct", "resolve", "retract", "confirm"):
                if not target or not self.store.get_claim(uid, target):
                    dropped.append({"op": name, "text": text, "reason": "missing/invalid target_claim_id"})
                    continue
            # --- time fields: legal + ordered ---
            vf = op.get("valid_from") if isinstance(op.get("valid_from"), str) else None
            vt = op.get("valid_to") if isinstance(op.get("valid_to"), str) else None
            event_at = op.get("event_at") if isinstance(op.get("event_at"), str) else None
            if event_at and parse_flexible(event_at) is None:
                dropped.append({"op": name, "text": text, "reason": "invalid event_at"})
                continue
            anchor = parse_iso(srcs[0]["message_at"])
            # v2.5: the source instant is NOT automatically the event instant. If
            # `event_at` merely repeats the source message's own time while the
            # source text states no clock, keep the DATE granularity (drop the
            # exact time) instead of promoting a known source time into an
            # unknown event time. Text and source timestamps are never changed.
            guards: list[str] = []
            event_at, event_guard = self._guard_event_at(
                event_at, anchor, [by_id[s["message_id"]]["text"] for s in srcs])
            if event_guard:
                guards.append(event_guard)
            # Ordering is validated on the RAW fields (a bare date counts as its own
            # start-of-day) BEFORE the exclusive-end expansion, so an inverted range
            # like from=06-02 to=06-01 is still rejected.
            if not self._range_ok(vf, vt):
                dropped.append({"op": name, "text": text, "reason": "invalid valid_from/valid_to"})
                continue
            # Unify the valid_from/valid_to contract in the SOURCE message's timezone:
            # valid_to is the EXCLUSIVE end instant. A truly bare date (YYYY-MM-DD)
            # means "valid through the whole of that day" and expands to the NEXT day
            # 00:00; a tz-aware datetime is an exact exclusive instant and is kept
            # verbatim, INCLUDING an exact 00:00 (an explicit midnight deadline is a
            # real instant, not a whole-day stand-in).
            vf = self._model_vf(vf, anchor)
            vt = self._model_vt(vt, anchor)
            relative = op.get("relative_phrase") if isinstance(op.get("relative_phrase"), str) else None
            resolved = None
            if relative:
                resolved = resolve_relative_phrase(relative, anchor)
                if resolved and not vf:
                    # tz-aware, anchored to the MESSAGE's own timezone (never a bare SH date)
                    vf = dt.datetime.combine(dt.date.fromisoformat(resolved), dt.time.min,
                                             anchor.tzinfo).isoformat()
            if kind in PLAN_KINDS and resolved and not vt:
                vt = dt.datetime.combine(dt.date.fromisoformat(resolved) + dt.timedelta(days=1),
                                         dt.time.min, anchor.tzinfo).isoformat()
            # A determinable period boundary (e.g. '月底') gives a RANGE END used
            # only to judge whether a plan/temporary/undecided window has elapsed;
            # it is never an exact action day, and an unresolved/ambiguous phrase
            # keeps its dependency/range untouched (valid_to stays null).
            bound_end_from_phrase = None
            if relative and kind in BOUND_KINDS and not vt:
                bound_end_from_phrase = resolve_bound_end(relative, anchor)
                if bound_end_from_phrase and not self._phrase_in_source(
                        relative, [by_id[s["message_id"]]["text"] for s in srcs]):
                    # The phrase must actually occur in the source; a calendar
                    # range the user never wrote stays unknown.
                    bound_end_from_phrase = None
                if bound_end_from_phrase:
                    # bare last-day date -> exclusive next-day 00:00 in source tz
                    vt = self._bare_day_end(bound_end_from_phrase, anchor)
            # RE-validate the final range (model-provided or engine-derived)
            if not self._range_ok(vf, vt):
                dropped.append({"op": name, "text": text,
                                "reason": "derived valid_from/valid_to failed validation"})
                continue
            # deterministic date guard on the claim text
            claim_dates = dates_in_text(text)
            if claim_dates:
                allowed: set[str] = set()
                for s in srcs:
                    anchor = parse_iso(s["message_at"])
                    allowed.add(anchor.date().isoformat())
                    allowed |= dates_in_text(by_id[s["message_id"]]["text"])
                    allowed |= weekday_dates_in_text(by_id[s["message_id"]]["text"], anchor)
                if resolved:
                    allowed.add(resolved)
                intruders = sorted(claim_dates - allowed)
                if intruders:
                    dropped.append({"op": name, "text": text,
                                    "reason": f"claim introduces non-derivable date(s): {intruders}"})
                    continue
            out.append({
                "op": name, "text": text, "kind": kind,
                "importance": (op.get("importance") or "medium"),
                "source_ids": sorted({s["message_id"] for s in srcs}),
                "sources": srcs,
                # Only meaningful in the bounded repair round: names the round-1
                # candidate (by engine-assigned id) this op repairs. Never guessed.
                "repair_of": (op.get("repair_of") if isinstance(op.get("repair_of"), str)
                              and op.get("repair_of").strip() else None),
                "relative_phrase": relative, "resolved_from_phrase": resolved,
                "bound_end_from_phrase": bound_end_from_phrase,
                "event_at": event_at, "valid_from": vf, "valid_to": vt,
                "target_claim_id": target, "resolution": resolution or None,
                "reason": (op.get("reason") or ""),
                "guards": guards,
            })
        return out

    @staticmethod
    def _guard_event_at(event_at: str | None, anchor: dt.datetime,
                        source_texts: list[str]) -> tuple[str | None, str | None]:
        """Keep "today" events at DATE granularity when no clock is stated.

        Scope (deliberately narrow): the guard fires ONLY when `event_at` is
        exactly the source message's own instant, that instant has a real
        time-of-day (not midnight), and none of the source texts contains clock
        evidence. In that case the model has lifted a known *source* time into an
        unsupported *event* time, so the exact time is dropped and the claim keeps
        its date granularity. Any explicit clock in the source — or any event_at
        that differs from the message instant — is left exactly as proposed.
        Returns (event_at, guard_note).
        """
        if not event_at or not isinstance(event_at, str):
            return event_at, None
        try:
            parsed = parse_iso(event_at)
        except TimeError:
            return event_at, None               # invalid values are handled elsewhere
        if parsed != anchor:
            return event_at, None               # not a copy of the source instant
        # The same instant may be written in ANY timezone offset. Day granularity
        # can only be judged in the SOURCE message's timezone: `00:00Z` is the
        # source's 08:00, not a date-only value, while a true source-local 00:00
        # already carries date granularity and is left alone.
        local = parsed.astimezone(anchor.tzinfo)
        if local.timetz().replace(tzinfo=None) == dt.time.min:
            return event_at, None               # already date-granular (in source tz)
        if any(_CLOCK_EVIDENCE.search(t or "") for t in source_texts):
            return event_at, None               # the source states a clock
        return None, ("event_at 等于来源消息瞬时且来源未给时刻证据："
                      "按日期精度保留，不把已知来源时间当作未知事件时间")

    def _backfill_bounds(self, uid: str, trace: dict) -> int:
        """Derive a missing valid_to from the claim's OWN sourced phrase.

        Only claims whose kind has an applicability window (plan / temporary_state
        / undecided), whose status is still open, and whose `valid_to` is empty but
        whose `relative_phrase` is a determinable calendar range are touched. The
        anchor is the claim's earliest SOURCE message time — never `now` — so the
        bound is derived from the material, and a claim with no source is skipped.
        The value written is the exclusive instant just after the range's last day
        (a SYSTEM upper bound, not the user's exact deadline).
        """
        derived: list[dict] = []
        for c in self.store.list_claims(uid):
            if c["kind"] not in BOUND_KINDS or c["status"] not in OPEN_STATUSES:
                continue
            if c.get("valid_to"):
                continue
            phrase = c.get("relative_phrase")
            if not phrase:
                continue
            sources = self.store.claim_source_messages(uid, c["claim_id"])
            if not sources:
                continue                      # no sourced anchor -> leave unknown
            try:
                anchor = parse_iso(sources[0]["message_at"])
            except TimeError:
                continue
            last_day = resolve_bound_end(phrase, anchor)
            if not last_day:
                continue                      # not a determinable calendar range
            if not self._phrase_in_source(phrase, [s.get("text") for s in sources]):
                continue                      # phrase not actually in the source
            valid_to = dt.datetime.combine(
                dt.date.fromisoformat(last_day) + dt.timedelta(days=1),
                dt.time.min, anchor.tzinfo).isoformat()
            self.store.set_claim_bound(
                uid, c["claim_id"], valid_to,
                note=(f"有效窗口上界由来源原话『{phrase}』推定（系统推定，非用户精确截止；"
                      f"锚点 {anchor.isoformat()}）"))
            self.store.add_event(uid, c["claim_id"], "bound_derived", c["status"], c["status"],
                                 f"valid_to={valid_to} from phrase {phrase!r} @ {anchor.isoformat()}")
            derived.append({"claim_id": c["claim_id"], "relative_phrase": phrase,
                            "anchor": anchor.isoformat(), "valid_to": valid_to,
                            "source": "engine:derived_from_source_phrase"})
        trace["bounds_backfilled"] = derived
        return len(derived)

    @staticmethod
    def _phrase_in_source(phrase: str | None, source_texts: list[str | None]) -> bool:
        """True only when the time phrase actually occurs in the source text.

        A derived calendar bound must be anchored in what the user actually said —
        never invented from `now` or from a topic name.
        """
        needle = _norm_text(phrase)
        if not needle:
            return False
        return any(needle in _norm_text(t) for t in source_texts)

    @staticmethod
    def _claim_note(c: dict) -> str | None:
        """Claim note, naming the source of an ENGINE-derived validity bound.

        The note is provenance only: the window is a SYSTEM upper bound derived
        from the user's own phrase, never the user's exact deadline.
        """
        note = c.get("reason") or ""
        if c.get("bound_end_from_phrase"):
            marker = (f"有效窗口上界由来源原话『{c.get('relative_phrase')}』推定"
                      f"（{c['bound_end_from_phrase']}，系统推定，非用户精确截止）")
            note = f"{note}；{marker}" if note else marker
        return note or None

    @staticmethod
    def _is_index(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    def _process_verdicts(self, candidates: list[dict], vres: Any, dropped: list[dict]) -> list[tuple[dict, dict]]:
        """Validate the verify result strictly, then apply the LEGAL verdicts.

        Malformed structure (non-object element, missing/invalid index or verdict,
        missing boolean flags, duplicate index, or indices that do not cover the
        candidates) fails closed. A well-formed `unsupported`/`partial`/flag-false
        verdict is a LEGAL rejection: the op is dropped and the batch still commits.
        """
        if not isinstance(vres, dict):
            raise ModelError("verify result is not a JSON object")
        raw = vres.get("verdicts")
        if raw is None:
            raise ModelError("verify result missing 'verdicts' array")
        if not isinstance(raw, list):
            raise ModelError("verify result 'verdicts' is not an array")
        verdicts: dict[int, dict] = {}
        for v in raw:
            if not isinstance(v, dict):
                raise ModelError("verdict element is not an object")
            index = v.get("index")
            if not self._is_index(index):
                raise ModelError("verdict element missing/invalid integer index")
            if v.get("verdict") not in ("supported", "partial", "unsupported"):
                raise ModelError(f"verdict element has invalid verdict: {v.get('verdict')!r}")
            for flag in ("role_ok", "time_ok", "overgeneralized"):
                if not isinstance(v.get(flag), bool):
                    raise ModelError(f"verdict element missing boolean '{flag}'")
            if index in verdicts:
                raise ModelError(f"duplicate verdict index {index}")
            verdicts[index] = v
        if set(verdicts) != set(range(len(candidates))):
            raise ModelError(f"verdict indices {sorted(verdicts)} do not cover all candidates")
        accepted: list[tuple[dict, dict]] = []
        for i, c in enumerate(candidates):
            v = verdicts[i]
            verdict = v["verdict"]
            flags_ok = v["role_ok"] is True and v["time_ok"] is True and v["overgeneralized"] is False
            # A replacement op (correct/retract) must state HOW it relates to the
            # target. Only an explicit "replaces" supersedes/retracts; "compatible"
            # keeps the target and records the new content independently; anything
            # missing/malformed (incl. a bare boolean `replacement_ok`) is treated
            # as "unknown" -> a pending relation with uncertainty.
            relation = self._relation_of(c, v)
            if verdict == "supported" and flags_ok:
                accepted.append((c, {**v, "_relation": relation}))
            elif verdict == "partial" and flags_ok and c["op"] in ("add", "correct"):
                accepted.append((c, {**v, "_unconfirmed": True, "_relation": relation}))
            else:
                dropped.append({"op": c["op"], "text": c["text"],
                                "reason": "verify:" + str(v.get("issue") or verdict)})
        return accepted

    @staticmethod
    def _relation_of(c: dict, v: dict) -> str:
        """Normalise the verifier's stated relation for a replacement op.

        A non-replacement op has no relation (returns ""). For correct/retract:
          - an explicit enum string in RELATION_VALUES is taken as-is;
          - a bare boolean `replacement_ok` is accepted only as True->"replaces",
            False->"unknown" (never silently "compatible");
          - everything else (missing, numbers, strings like "true", null) -> "unknown".
        """
        if c["op"] not in RELATION_OPS:
            return ""
        raw = v.get("relation")
        if isinstance(raw, str) and raw.strip().lower() in RELATION_VALUES:
            return raw.strip().lower()
        legacy = v.get("replacement_ok")
        if legacy is True:
            return RELATION_REPLACES
        if legacy is False:
            return RELATION_UNKNOWN
        return RELATION_UNKNOWN

    # ---- bounded repair (v2.4 candidate) ---------------------------------
    @staticmethod
    def _op_full_key(c: dict) -> tuple:
        """Exact identity INCLUDING the substantive fields.

        Two ops are the "same candidate" only when text AND every substantive
        field (op, sources, target, kind, relative phrase, valid_from/valid_to,
        event_at, resolution) match. A repair that only changes a time/kind field
        therefore has a DIFFERENT key and still reaches an independent verify.
        """
        return (_norm_text(c.get("text")), c["op"], tuple(sorted(c.get("source_ids") or [])),
                c.get("target_claim_id"), c.get("kind"), _norm_text(c.get("relative_phrase")),
                c.get("valid_from"), c.get("valid_to"), c.get("event_at"), c.get("resolution"))

    @staticmethod
    def _brief(c: dict, v: dict | None = None) -> dict:
        out = {"op": c["op"], "text": c["text"], "kind": c["kind"],
               "source_ids": c.get("source_ids"), "target_claim_id": c.get("target_claim_id")}
        if v is not None:
            out["verdict"] = v.get("verdict")
            out["issue"] = v.get("issue")
        return out

    def _repair_round(self, uid: str, now: Any, payload: dict, new_msgs: list[dict],
                      new_by_id: dict[str, dict], accepted: list[tuple[dict, dict]],
                      dropped: list[dict], model_calls: dict, trace: dict) -> dict:
        """EXACTLY ONE bounded re-proposal + independent re-verify.

        Triggered when the first round dropped an item or accepted one only as
        unconfirmed (`partial`). The model re-proposes from the SAME original
        evidence plus the specific rejection/verdict feedback; the repaired
        candidates re-run the full deterministic checks and a fresh, independent
        verify. `corrected_text` is never adopted directly and a `partial` is
        never auto-promoted — a repaired item still has to earn `supported`.
        """
        info: dict[str, Any] = {"triggered": False, "rounds": 0, "feedback": [],
                                "replaced": [], "accepted": [], "dropped": []}
        partial = [(c, v) for c, v in accepted if v.get("_unconfirmed")]
        if MAX_REPAIR_ROUNDS < 1 or not (dropped or partial):
            trace["repair"] = info
            return {"accepted": accepted}

        info["triggered"] = True
        info["rounds"] = 1
        # Build the round-1 repair units and hand each a STABLE engine-assigned id.
        # The model can only repair a unit by echoing its id in `repair_of`, so a
        # repair maps to exactly ONE unit and cannot silently remove another.
        units: list[dict] = []
        for d in dropped:
            units.append({"id": f"cand-{len(units) + 1}", "kind": "dropped", "op": d.get("op"),
                          "text": d.get("text"), "accepted_index": None,
                          "reason": d.get("reason")})
        for idx, (c, v) in enumerate(accepted):
            if not v.get("_unconfirmed"):
                continue
            units.append({"id": f"cand-{len(units) + 1}", "kind": "partial", "op": c["op"],
                          "text": c["text"], "accepted_index": idx,
                          "reason": "verify:" + str(v.get("issue") or v.get("verdict")),
                          "note": "方向正确但仅判为未确认（partial）；请只做最小修正"})
        feedback = [{"candidate_id": u["id"], "op": u["op"], "text": u["text"],
                     "reason": u["reason"], **({"note": u["note"]} if u.get("note") else {})}
                    for u in units]
        unit_by_id = {u["id"]: u for u in units}
        info["feedback"] = feedback
        repair_payload = {**payload, "previous_attempt": {
            "rejected": feedback,
            "instruction": ("只针对上面列出的被拒/未确认条目做最小修正后重新给出 operations："
                            "要修正哪一条，就在该 op 里写 repair_of=<那条的 candidate_id>；"
                            "保留仍然成立的核心信息，去掉没有来源支持的修饰；不要把整条丢掉；"
                            "不要新增原消息里没有的人/事/时间；不要重复已经通过的条目。")}}
        trace["repair_input"] = repair_payload
        prop2 = self.model.propose(self.prompts["propose"], repair_payload)
        model_calls["propose"] += 1
        trace["repair_response"] = prop2
        if not isinstance(prop2, dict):
            raise ModelError("repair propose result is not a JSON object")

        repair_dropped: list[dict] = []
        cands2 = self._check_ops(uid, prop2.get("operations"), new_msgs, repair_dropped)
        # Dedup by exact identity (incl. substantive fields) against round-1
        # accepted items AND within this round. An exact repeat of a round-1
        # partial is NOT re-verified (stays unconfirmed, never auto-promoted); a
        # repair that changes any substantive field has a different key and does
        # reach an independent verify.
        seen = {self._op_full_key(c) for c, _ in accepted}
        deduped: list[dict] = []
        for c in cands2:
            key = self._op_full_key(c)
            if key in seen:
                info.setdefault("deduped", []).append(self._brief(c))
                continue
            seen.add(key)
            deduped.append(c)
        cands2 = deduped
        info["candidates"] = [self._brief(c) for c in cands2]
        if not cands2:
            dropped.extend(repair_dropped)
            info["dropped"] = [{**d} for d in repair_dropped]
            trace["repair"] = info
            return {"accepted": accepted}

        vpayload2 = {"now": now.isoformat(),
                     "claims": [self._verify_item(uid, i, c, new_by_id)
                                for i, c in enumerate(cands2)]}
        trace["repair_verify_input"] = vpayload2
        vres2 = self.model.verify(self.prompts["verify"], vpayload2)
        model_calls["verify"] += 1
        trace["repair_verify_response"] = vres2
        accepted2 = self._process_verdicts(cands2, vres2, repair_dropped)
        dropped.extend(repair_dropped)
        info["accepted"] = [self._brief(c, v) for c, v in accepted2]
        info["dropped"] = [{**d} for d in repair_dropped]

        # Apply the ONE-TO-ONE repair map: an accepted round-2 item that names a
        # round-1 `partial` unit via `repair_of` replaces exactly that unit. A
        # round-1 partial that no round-2 item named is kept as it was; a round-2
        # item with no/unknown `repair_of` is simply added (nothing is deleted).
        consumed: set[int] = set()
        removed_indices: set[int] = set()
        info["unmatched_repair_of"] = []
        for c2, _v2 in accepted2:
            ref = c2.get("repair_of")
            unit = unit_by_id.get(ref) if ref else None
            if unit is None:
                if ref:
                    info["unmatched_repair_of"].append(ref)
                continue
            if unit["kind"] == "partial" and unit["accepted_index"] not in consumed:
                consumed.add(unit["accepted_index"])
                removed_indices.add(unit["accepted_index"])
                info["replaced"].append(unit["text"])
        final = [(c, v) for i, (c, v) in enumerate(accepted) if i not in removed_indices]
        final += accepted2
        trace["repair"] = info
        return {"accepted": final}

    # ---- source-citation resolvability (v2.4 candidate) -------------------
    def _valid_source_ids(self, uid: str) -> set[str]:
        """Every id a summary is allowed to cite: real message ids + claim ids."""
        ids = {m["message_id"] for m in self.store.messages(uid)}
        ids |= {c["claim_id"] for c in self.store.list_claims(uid)}
        return ids

    def _source_literals(self, uid: str, material: str) -> set[str]:
        """Bracketed tokens that are literally quoted from the material/source.

        A bracketed token that occurs verbatim (without its brackets) in the
        provided material or in any stored source message is source content —
        e.g. a business model/part number like [AB12], or a bracketed list that is
        part of the original text — NOT a citation. This is a general "quoted from
        input" rule, not a whitelist of words or models.
        """
        haystack = [material or ""]
        haystack += [m["text"] or "" for m in self.store.messages(uid)]
        haystack += [c["text"] or "" for c in self.store.list_claims(uid)]
        return {tok.strip() for tok in _BRACKETED.findall(" ".join(haystack)) if tok.strip()}

    @classmethod
    def _citation_parts(cls, token: str) -> list[str]:
        """Non-empty parts of one bracketed token, split on citation separators."""
        return [p for p in (s.strip() for s in _CITATION_SEPARATORS.split(token)) if p]

    def _invalid_citations(self, uid: str, text: str, material: str) -> tuple[list[str], list[str]]:
        """(unknown_ids, malformed_tokens) for one summary.

        Digit presence is NOT required: `source` is a valid citation target.
        A combined form (`[id-a; id-b]`, `[id-a, missing-source]`) is forbidden,
        but it is NEVER skipped: each id-shaped part is validated against the real
        message/claim ids, so an unknown id inside a combined bracket is still
        reported. A bracketed token that appears VERBATIM in the material/source
        is quoted source text, not a citation, and is exempt as a whole.
        """
        valid = self._valid_source_ids(uid)
        literals = self._source_literals(uid, material)
        unknown: list[str] = []
        malformed: list[str] = []
        seen: set[str] = set()
        for tok in _BRACKETED.findall(text or ""):
            tok = tok.strip()
            if not tok or tok in literals:
                continue
            parts = self._citation_parts(tok)
            shaped = [p for p in parts if _CITATION_SHAPE.match(p)]
            if not shaped:
                continue                       # CJK words / dates / prose
            combined = not (len(parts) == 1 and len(shaped) == 1)
            if combined:
                malformed.append(tok)
            for cid in shaped:
                if cid in seen or cid in valid or cid in literals:
                    continue
                seen.add(cid)
                unknown.append(cid)
        return unknown, malformed

    @staticmethod
    def _combined_citations(unknown: list[str], malformed: list[str]) -> list[str]:
        """Flat, order-preserving list of everything that blocks publication."""
        out = list(unknown)
        for tok in malformed:
            if tok not in out:
                out.append(tok)
        return out

    def _check_citations(self, uid: str, now: Any, material: str, summary_md: str,
                         model_calls: dict, trace: dict) -> str:
        """Reject a summary whose citations do not resolve, or use a forbidden form.

        Two things block publication: (1) an id that is neither a known message id
        nor a known claim id, nor quoted verbatim from the input; (2) a combined
        bracket such as `[a; b]` / `[a, b]`, which the protocol forbids. Exactly
        ONE constrained regeneration is allowed (feedback = the offending ids and
        combined forms). If the regenerated summary still violates the contract,
        this fails closed: the batch is rolled back, the previous complete version
        is kept, and the error is visible. Ids are never guessed or rewritten.
        """
        unknown, malformed = self._invalid_citations(uid, summary_md, material)
        invalid = self._combined_citations(unknown, malformed)
        trace["summary_citations"] = {"invalid": invalid, "unknown": unknown,
                                      "malformed": malformed, "regenerated": False}
        if not invalid:
            return summary_md
        if MAX_SUMMARY_REGEN < 1:
            raise ModelError(f"summary cites unresolvable or forbidden citation(s): {invalid}")
        regen_payload = {
            "now": now.isoformat(), "memory_material": material,
            "invalid_source_ids": invalid,
            "forbidden_combined_citations": malformed,
            "instruction": ("上次回复里这些引用不可用：" + "、".join(invalid) +
                            "。只能用材料里真实出现的消息 id 或 claim id（照抄，不要改写或编造）；"
                            "不确定就删掉该引用。**每个方括号里只写一个 id**，不要用 [a; b] 或 "
                            "[a, b] 这类合并写法；多个来源要写成 [a] [b]。其余内容保持不变。"),
        }
        trace["summary_regen_input"] = regen_payload
        try:
            summary2 = self.model.summarize(self.prompts["summarize"], regen_payload)
            model_calls["summarize"] += 1
        except BaseException as exc:  # noqa: BLE001
            raise ModelError(f"summarize(regenerate): {type(exc).__name__}: {exc}") from exc
        trace["summary_regen_response"] = summary2
        trace["summary_citations"]["regenerated"] = True
        if not (summary2 or "").strip():
            raise ModelError("summary regeneration returned empty content")
        unknown2, malformed2 = self._invalid_citations(uid, summary2, material)
        invalid2 = self._combined_citations(unknown2, malformed2)
        trace["summary_citations"]["unknown_after_regen"] = unknown2
        trace["summary_citations"]["malformed_after_regen"] = malformed2
        trace["summary_citations"]["invalid_after_regen"] = invalid2
        if invalid2:
            raise ModelError("summary still has unresolvable or forbidden citation(s) after one "
                             f"regeneration: {invalid2}")
        return summary2

    def _verify_item(self, uid: str, index: int, c: dict, new_by_id: dict[str, dict]) -> dict:
        item = {
            "index": index, "op": c["op"], "text": c["text"], "kind": c["kind"],
            "importance": c["importance"], "relative_phrase": c["relative_phrase"],
            "resolved_from_phrase": c["resolved_from_phrase"],
            "bound_end_from_phrase": c.get("bound_end_from_phrase"),
            "event_at": c["event_at"], "valid_from": c["valid_from"], "valid_to": c["valid_to"],
            "resolution": c.get("resolution"), "target_claim_id": c.get("target_claim_id"),
            "evidence": [
                {"message_id": s["message_id"], "role": s["role"], "message_at": s["message_at"],
                 "text": new_by_id[s["message_id"]]["text"], "quote": s["quote"]}
                for s in c["sources"]
            ],
        }
        target = c.get("target_claim_id")
        if target:
            item["target_claim"] = self._target_view(uid, target)
            item["target_evidence"] = self.store.claim_source_messages(uid, target)
        return item

    def _status_for(self, kind: str) -> str:
        if kind == "plan":
            return "planned"
        if kind == "temporary_state":
            return "temporary"
        return "active"

    @staticmethod
    def _range_ok(vf: str | None, vt: str | None) -> bool:
        """A valid range: each side parseable (if present) and start <= end."""
        start = parse_flexible(vf) if vf else None
        end = parse_flexible(vt) if vt else None
        if vf and start is None:
            return False
        if vt and end is None:
            return False
        if start and end and start > end:
            return False
        return True

    @staticmethod
    def _bare_day_end(day_iso: str, anchor: dt.datetime) -> str:
        """Exclusive end instant for a whole-day boundary: next day 00:00 (source tz)."""
        return dt.datetime.combine(dt.date.fromisoformat(day_iso) + dt.timedelta(days=1),
                                   dt.time.min, anchor.tzinfo).isoformat()

    @classmethod
    def _model_vf(cls, value: str | None, anchor: dt.datetime) -> str | None:
        """Normalise a MODEL-provided valid_from into a source-timezone instant.

        A bare date means the start of that day (inclusive).
        """
        if not value:
            return value
        try:
            day = dt.date.fromisoformat(value)
        except ValueError:
            return value                      # already a tz-aware datetime (validated)
        return dt.datetime.combine(day, dt.time.min, anchor.tzinfo).isoformat()

    @staticmethod
    def _model_vt(value: str | None, anchor: dt.datetime) -> str | None:
        """Normalise a MODEL-provided valid_to into the EXCLUSIVE end instant.

        valid_to is the instant at which the window has ended. Only a truly BARE
        date (YYYY-MM-DD) is day-granular: it means "valid through the whole of that
        day", so it expands to the NEXT day 00:00 in the SOURCE timezone. Any
        tz-aware datetime is an exact exclusive instant the model declared and is
        kept verbatim — including one pinned to exactly 00:00. (The engine's own
        derived boundaries already emit an exclusive instant and never re-enter
        here, so they are not re-shifted.)
        """
        if not value:
            return value
        try:
            day = dt.date.fromisoformat(value)
        except ValueError:
            return value                      # tz-aware datetime (validated) kept verbatim
        return dt.datetime.combine(day + dt.timedelta(days=1), dt.time.min,
                                   anchor.tzinfo).isoformat()

    def _apply_op(self, uid: str, c: dict, v: dict) -> str | None:
        op = c["op"]
        unconfirmed = bool(v.get("_unconfirmed"))
        text = c["text"]  # never auto-rewrite from corrected_text (no edit bypass)
        if v.get("corrected_text"):
            self.store.diagnose(uid, {"type": "corrected_text_advisory", "op": op,
                                      "original": c["text"], "corrected_text": v["corrected_text"]})
        status = "unconfirmed" if unconfirmed else self._status_for(c["kind"])
        version = self.store.current_version(uid)
        claim = {
            "text": text, "kind": c["kind"], "status": status, "importance": c["importance"],
            "valid_from": c["valid_from"], "valid_to": c["valid_to"],
            "relative_phrase": c["relative_phrase"], "resolved_from_phrase": c["resolved_from_phrase"],
            "event_at": c["event_at"],
            "verification": "model:" + str(v.get("verdict") or "supported"),
            "note": self._claim_note(c),
        }
        if op == "add":
            cid = self.store.insert_claim(uid, claim, c["sources"], version)
            self.store.add_event(uid, cid, "created", None, status, c.get("reason") or "")
            return cid
        if op == "correct":
            target = c.get("target_claim_id")
            relation = v.get("_relation") or RELATION_UNKNOWN
            target_claim = self.store.get_claim(uid, target) if target else None
            if relation == RELATION_COMPATIBLE:
                # The verifier proved the new statement can hold TOGETHER with the
                # target (a clarification, or a denial of someone else's reading):
                # keep the target and record the new content as its own claim.
                cid = self.store.insert_claim(uid, claim, c["sources"], version)
                self.store.add_event(uid, cid, "created", None, status,
                                     "compatible statement kept as new claim")
                return cid
            if relation == RELATION_UNKNOWN:
                # No proven relation: keep the PENDING link (new claim -> target)
                # and the uncertainty, never two independent confirmed facts. The
                # unproven edge is stored DURABLY (`edge='unknown'`) so a later
                # confirm of the new CONTENT can never turn it into a proven
                # replacement.
                weak = {**claim, "replaces": target, "status": "unconfirmed", "edge": "unknown",
                        "note": "替换关系未证明，待核验"}
                cid = self.store.insert_claim(uid, weak, c["sources"], version)
                if target_claim:
                    self.store.set_claim_status(uid, target, "contested",
                                                note=f"替换关系未证明（{cid} 指向本主张）")
                    self.store.add_event(uid, target, "contested", target_claim["status"],
                                         "contested", f"unproven replacement by {cid}")
                self.store.add_event(uid, cid, "created", None, "unconfirmed",
                                     "unproven replacement relation")
                self.store.diagnose(uid, {"type": "replacement_not_proven", "op": op,
                                          "target_claim_id": target, "text": text})
                return cid
            new_id = self.store.insert_claim(uid, {**claim, "replaces": target, "edge": "replaces"},
                                             c["sources"], version)
            if target_claim and not unconfirmed:
                self.store.set_claim_status(uid, target, "superseded", replaced_by=new_id,
                                            note=f"superseded by {new_id}")
                self.store.add_event(uid, target, "superseded", None, "superseded", f"by {new_id}")
                self.store.add_event(uid, new_id, "created", "superseded", status, f"corrects {target}")
            elif target_claim and unconfirmed:
                # A weak (partial) correction must NOT leave the old claim looking
                # settled: mark it contested so the current state is not over-claimed.
                self.store.set_claim_status(uid, target, "contested",
                                            note=f"contested by unconfirmed {new_id}")
                self.store.add_event(uid, target, "contested", target_claim["status"], "contested",
                                     f"weak correction by {new_id}")
                self.store.add_event(uid, new_id, "created", "contested", status, f"weak correction of {target}")
            else:
                self.store.add_event(uid, new_id, "created", None, status,
                                     "unconfirmed correction" if unconfirmed else f"corrects {target}")
            return new_id
        if unconfirmed:
            return None  # weak evidence must not change an existing claim
        target = c.get("target_claim_id")
        target_claim = self.store.get_claim(uid, target) if target else None
        if not target_claim:
            return None
        if op == "resolve":
            if target_claim["kind"] == "undecided":
                # "undecided -> decided" is a state CHANGE, not a plan finishing.
                # The verified new statement (its own text/kind/sources) becomes
                # the current fact; the old undecided claim keeps its own text and
                # sources as sourced history. Flattening the target's status to
                # `completed` would leave the material showing the OLD undecided
                # sentence as a completed item and lose the decided content.
                new_id = self.store.insert_claim(
                    uid, {**claim, "replaces": target, "edge": RELATION_REPLACES},
                    c["sources"], version)
                self.store.set_claim_status(uid, target, "superseded", replaced_by=new_id,
                                            note=f"未定状态已被新的事实确定取代（{new_id}）")
                for s in c["sources"]:
                    self.store.add_source(target, s)   # history keeps the deciding evidence
                self.store.add_event(uid, target, "superseded", target_claim["status"],
                                     "superseded", f"settled by {new_id}")
                self.store.add_event(uid, new_id, "created", "superseded", status,
                                     f"settles undecided {target}")
                return new_id
            new_status = {"completed": "completed", "abandoned": "abandoned",
                          "in_progress": "in_progress"}.get(c.get("resolution"), "in_progress")
            self.store.set_claim_status(uid, target, new_status, note=c.get("reason") or None)
            for s in c["sources"]:
                self.store.add_source(target, s)
            self.store.add_event(uid, target, "resolved", target_claim["status"], new_status,
                                 f"evidence {c['source_ids']}")
            return target
        if op == "retract":
            relation = v.get("_relation") or RELATION_UNKNOWN
            if relation == RELATION_COMPATIBLE:
                # Same rule as `correct`: a proven-compatible statement does not
                # retract; keep the target and record the statement on its own.
                cid = self.store.insert_claim(uid, claim, c["sources"], version)
                self.store.add_event(uid, cid, "created", None, status,
                                     "compatible statement kept as new claim")
                return cid
            if relation == RELATION_UNKNOWN:
                # No proven relation: keep the pending link + uncertainty instead
                # of leaving the target looking settled or silently retracting.
                # Durable `edge='unknown'` (see the `correct` branch).
                weak = {**claim, "replaces": target, "status": "unconfirmed", "edge": "unknown",
                        "note": "撤回关系未证明，待核验"}
                cid = self.store.insert_claim(uid, weak, c["sources"], version)
                self.store.set_claim_status(uid, target, "contested",
                                            note=f"撤回关系未证明（{cid} 指向本主张）")
                self.store.add_event(uid, target, "contested", target_claim["status"],
                                     "contested", f"unproven retraction by {cid}")
                self.store.add_event(uid, cid, "created", None, "unconfirmed",
                                     "unproven retraction relation")
                self.store.diagnose(uid, {"type": "replacement_not_proven", "op": op,
                                          "target_claim_id": target, "text": text})
                return cid
            self.store.set_claim_status(uid, target, "retracted", note=c.get("reason") or None)
            for s in c["sources"]:
                self.store.add_source(target, s)
            self.store.add_event(uid, target, "retracted", target_claim["status"], "retracted",
                                 c.get("reason") or "")
            return target
        if op == "confirm":
            if target_claim["status"] not in ("unconfirmed", "active", "planned", "temporary",
                                              "expired", "historical", "contested", "in_progress"):
                return None
            new_status = self._status_for(target_claim["kind"])
            prior = target_claim.get("verification") or "unknown"
            # The edge that links this claim to an older one (if any) is stored
            # durably on the claim. Confirming the CONTENT must never upgrade an
            # edge that was never proven: only `edge == "replaces"` closes the
            # chain. An `unknown` edge is confirmed as CONTENT only, and the old
            # target stays `contested` (the relation is still unknown).
            edge = target_claim.get("edge")
            chain_closable = edge == "replaces"
            note = "confirmed" if chain_closable or edge is None else "内容已确认；替换关系仍未证明"
            self.store.set_claim_status(uid, target, new_status, note=note,
                                        verification=f"model:supported(confirmed; prior {prior})")
            for s in c["sources"]:
                self.store.add_source(target, s)
            self.store.add_event(uid, target, "confirmed", target_claim["status"], new_status,
                                 f"prior verification {prior}; edge={edge or 'none'}")
            # Close the replacement chain in the SAME transaction: if this claim
            # REPLACED an older one (edge proven), the older 'contested' verdict
            # from a weak (partial) correction is resolved (superseded by the
            # now-confirmed claim). Prior partial history stays in the event log /
            # verification string. An unproven edge leaves the old target as it
            # is, so a compatible statement can never silently supersede it.
            old_id = target_claim.get("replaces")
            old = self.store.get_claim(uid, old_id) if old_id else None
            if old and old["status"] == "contested" and chain_closable:
                self.store.set_claim_status(uid, old_id, "superseded", replaced_by=target,
                                            note=f"superseded by {target} (confirmed)")
                self.store.add_event(uid, old_id, "resolved_contested", "contested", "superseded",
                                     f"weak correction confirmed by {target}")
            elif old and old["status"] == "contested":
                # Keep the pending relation visible and unresolved.
                self.store.add_event(uid, old_id, "edge_unresolved", "contested", "contested",
                                     f"content of {target} confirmed but replacement relation "
                                     f"is unproven (edge={edge or 'unknown'})")
            return target
        return None

    def _expire(self, uid: str, c: dict) -> None:
        if c["status"] == "unconfirmed":
            # Time passing must NOT raise confidence. Keep the claim unconfirmed,
            # record the elapsed window once, and leave later explicit confirmation
            # legal (confirm() restores the status for the claim's kind).
            note = f"有效窗口 {c['valid_to']} 已过，仍未确认（核验 {c.get('verification') or 'n/a'}）"
            self.store.set_claim_status(uid, c["claim_id"], "unconfirmed", note=note)
            self.store.add_event(uid, c["claim_id"], "window_elapsed", "unconfirmed", "unconfirmed",
                                 "valid_to passed but claim remains unconfirmed")
            return
        if c["kind"] == "undecided":
            # An elapsed determinable window must NOT settle an open question:
            # keep it visible as still-undecided and record the window once.
            note = f"有效窗口 {c['valid_to']} 已过，仍未定（核验 {c.get('verification') or 'n/a'}）"
            self.store.set_claim_status(uid, c["claim_id"], c["status"], note=note)
            self.store.add_event(uid, c["claim_id"], "window_elapsed", c["status"], c["status"],
                                 "determinable window passed but item remains undecided")
            return
        if c["kind"] == "plan":
            new_status, detail = "plan_due_unresolved", "valid_to passed without completion evidence"
        elif c["kind"] == "temporary_state":
            new_status, detail = "expired", "temporary state lapsed"
        else:
            new_status, detail = "expired", "valid_to passed"
        self.store.set_claim_status(uid, c["claim_id"], new_status,
                                    note=f"{detail} (valid_to {c['valid_to']})")
        self.store.add_event(uid, c["claim_id"], "expired", c["status"], new_status, detail)
