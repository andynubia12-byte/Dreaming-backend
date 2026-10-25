"""v2.6 regressions: the "undecided -> decided" lifecycle (offline, no API).

A supported `resolve` on an `undecided` target must persist the VERIFIED new
statement (its own text / kind / sources) as the current fact, and turn the old
undecided claim into sourced history. Flattening the target's status onto the old
sentence loses the decision; ordinary plan resolve semantics must stay intact.

Generic synthetic fixtures only; no personal data.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from memmaint.engine import Engine  # noqa: E402
from memmaint.model import ModelError, ScriptedModel  # noqa: E402
from memmaint.prompts import load_prompts  # noqa: E402
from memmaint.store import Store  # noqa: E402

PROMPTS = load_prompts()
T1 = "2026-03-01T12:00:00+08:00"
T2 = "2026-03-02T12:00:00+08:00"

UNDECIDED = "用户尚未决定是否接那单活儿"
DECIDED = "用户接下了那单活儿"


def batch(uid, bid, now, msgs):
    return {"user_id": uid, "batch_id": bid, "now": now, "messages": msgs}


def msg(mid, text, at=T1, role="user"):
    return {"id": mid, "conversation_id": "c1", "role": role, "message_at": at, "text": text}


def op(text, mid, quote=None, kind="user_fact", **kw):
    base = {"op": "add", "text": text, "kind": kind, "importance": "medium",
            "evidence_message_ids": [mid], "quotes": [{"message_id": mid, "quote": quote or text}],
            "target_claim_id": None, "event_at": None, "relative_phrase": None,
            "valid_from": None, "valid_to": None, "reason": ""}
    base.update(kw)
    return base


def verdicts(payload, verdict="supported", **kw):
    return {"verdicts": [{"index": c["index"], "verdict": verdict, "role_ok": True, "time_ok": True,
                          "overgeneralized": False, "issue": "", "corrected_text": None, **kw}
                         for c in payload["claims"]]}


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.state = Path(self._tmp.name) / "state"
        self.store = Store(self.state)

    def tearDown(self):
        self._tmp.cleanup()

    def engine(self, propose=None, verify=None, summarize=None, store=None):
        return Engine(store or self.store, ScriptedModel(propose=propose, verify=verify,
                                                         summarize=summarize), PROMPTS)

    def seed_undecided(self, text=UNDECIDED, mid="m1", at=T1, now=T1):
        self.engine(propose=lambda s, p: {"operations": [op(text, mid, kind="undecided")]}).run_batch(
            batch("u", "b-seed", now, [msg(mid, text, at)]))
        return self.store.list_claims("u")[0]["claim_id"]

    def reopen(self):
        self.store.close()
        self.store = Store(self.state)
        return self.store

    def claim_texts(self):
        return {c["text"]: c for c in self.store.list_claims("u")}


class UndecidedToDecidedTests(Base):
    def _decide(self, target_id, decided=DECIDED, source="那单我接了", mid="m2",
                resolution="completed", verdict="supported", **vkw):
        prop = lambda s, p: {"operations": [
            op(decided, mid, quote=source, kind="user_fact", target_claim_id=target_id,
               op="resolve", resolution=resolution, importance="high")]}
        verify = (lambda s, p: verdicts(p, verdict, **vkw)) if verdict != "supported" else None
        return self.engine(propose=prop, verify=verify).run_batch(
            batch("u", "b-decide", T2, [msg(mid, source, T2)]))

    def test_decided_fact_is_persisted_as_current(self):
        old = self.seed_undecided()
        r = self._decide(old)
        self.assertEqual(r["status"], "ok")
        claims = self.claim_texts()
        new = claims[DECIDED]
        self.assertEqual(new["status"], "active")
        self.assertEqual(new["kind"], "user_fact")
        self.assertEqual(new["importance"], "high")
        self.assertEqual(new["text"], DECIDED)
        self.assertEqual(new["replaces"], old)

    def test_old_undecided_becomes_sourced_history(self):
        old = self.seed_undecided()
        self._decide(old)
        previous = self.store.get_claim("u", old)
        self.assertEqual(previous["status"], "superseded")
        self.assertIsNotNone(previous["replaced_by"])
        self.assertEqual(previous["text"], UNDECIDED)
        self.assertEqual(previous["kind"], "undecided")
        self.assertNotEqual(previous["status"], "completed")
        sources = {s["message_id"] for s in self.store.claim_source_messages("u", old)}
        self.assertIn("m2", sources)

    def test_material_shows_current_fact_and_history(self):
        old = self.seed_undecided()
        self._decide(old)
        material = self.store.memory_path("u").read_text("utf-8")
        self.assertIn(f"[active][user_fact][high] {DECIDED}", material)
        self.assertIn("[superseded][undecided]", material)
        self.assertNotIn("[completed][undecided]", material)
        current, history = material.split("## 历史与已变更")
        self.assertIn(DECIDED, current)
        self.assertIn(UNDECIDED, history)
        self.assertNotIn(UNDECIDED, current)

    def test_summary_input_does_not_read_as_still_undecided(self):
        old = self.seed_undecided()
        seen = {}
        prop = lambda s, p: {"operations": [
            op(DECIDED, "m2", quote="那单我接了", kind="user_fact",
               target_claim_id=old, op="resolve", resolution="completed", importance="high")]}
        self.engine(propose=prop,
                    summarize=lambda s, p: seen.setdefault("material", p["memory_material"]) or "x"
                    ).run_batch(batch("u", "b-decide", T2, [msg("m2", "那单我接了", T2)]))
        current = seen["material"].split("## 历史与已变更")[0]
        self.assertIn(DECIDED, current)
        self.assertNotIn(UNDECIDED, current)

    def test_decision_survives_reopen(self):
        old = self.seed_undecided()
        self._decide(old)
        self.reopen()
        claims = self.claim_texts()
        self.assertEqual(claims[DECIDED]["status"], "active")
        self.assertEqual(self.store.get_claim("u", old)["status"], "superseded")
        self.assertIn(f"[active][user_fact][high] {DECIDED}",
                      self.store.memory_path("u").read_text("utf-8"))

    def test_denial_decision_also_persists_the_decided_statement(self):
        old = self.seed_undecided()
        decided = "用户决定不接那单活儿"
        prop = lambda s, p: {"operations": [
            op(decided, "m3", quote="那单我不接了", kind="user_fact",
               target_claim_id=old, op="resolve", resolution="abandoned")]}
        self.engine(propose=prop).run_batch(batch("u", "b3", T2, [msg("m3", "那单我不接了", T2)]))
        claims = self.claim_texts()
        self.assertEqual(claims[decided]["status"], "active")
        self.assertEqual(claims[decided]["kind"], "user_fact")
        self.assertEqual(self.store.get_claim("u", old)["status"], "superseded")

    def test_undecided_still_open_keeps_open_kind(self):
        old = self.seed_undecided()
        prop = lambda s, p: {"operations": [
            op("用户仍在考虑是否接那单活儿", "m4", quote="我还在考虑那单", kind="undecided",
               target_claim_id=old, op="resolve", resolution="in_progress")]}
        self.engine(propose=prop).run_batch(batch("u", "b4", T2, [msg("m4", "我还在考虑那单", T2)]))
        new = self.claim_texts()["用户仍在考虑是否接那单活儿"]
        self.assertEqual(new["kind"], "undecided")
        self.assertEqual(new["status"], "active")
        self.assertEqual(self.store.get_claim("u", old)["status"], "superseded")


class ResolveVerdictTests(Base):
    """partial / unsupported / flag-false resolves must never be auto-accepted."""

    def _attempt(self, verdict, **vkw):
        old = self.seed_undecided()
        prop = lambda s, p: {"operations": [
            op(DECIDED, "m2", quote="那单我接了", kind="user_fact",
               target_claim_id=old, op="resolve", resolution="completed")]}
        verify = lambda s, p: verdicts(p, verdict, **vkw)
        r = self.engine(propose=prop, verify=verify).run_batch(
            batch("u", "b-decide", T2, [msg("m2", "那单我接了", T2)]))
        return old, r

    def test_partial_resolve_is_not_applied(self):
        old, r = self._attempt("partial", issue="证据只够部分支持")
        self.assertNotIn(DECIDED, self.claim_texts())
        self.assertEqual(self.store.get_claim("u", old)["status"], "active")
        self.assertEqual(self.store.get_claim("u", old)["kind"], "undecided")
        self.assertTrue(any("证据只够部分支持" in d.get("reason", "") for d in r["dropped_ops"]))

    def test_unsupported_resolve_is_not_applied(self):
        old, _r = self._attempt("unsupported", issue="证据不支持")
        self.assertNotIn(DECIDED, self.claim_texts())
        self.assertEqual(self.store.get_claim("u", old)["status"], "active")

    def test_role_or_time_flag_blocks_resolve(self):
        for flag in ("role_ok", "time_ok"):
            with self.subTest(flag=flag):
                self.setUp()
                try:
                    old, _r = self._attempt("supported", **{flag: False})
                    self.assertNotIn(DECIDED, self.claim_texts())
                    self.assertEqual(self.store.get_claim("u", old)["status"], "active")
                finally:
                    self.tearDown()

    def test_overgeneralized_blocks_resolve(self):
        old, _r = self._attempt("supported", overgeneralized=True)
        self.assertNotIn(DECIDED, self.claim_texts())
        self.assertEqual(self.store.get_claim("u", old)["status"], "active")


class PlanResolveUnchangedTests(Base):
    """Ordinary plan resolve semantics and history are preserved."""

    PLAN = "用户计划整理画册"

    def _plan(self, resolution):
        self.engine(propose=lambda s, p: {"operations": [op(self.PLAN, "m1", kind="plan")]}).run_batch(
            batch("u", "b1", T1, [msg("m1", self.PLAN)]))
        cid = self.store.list_claims("u")[0]["claim_id"]
        prop2 = lambda s, p: {"operations": [
            op("用户整理画册的情况更新", "m2", quote="画册的事有进展", kind="plan",
               op="resolve", target_claim_id=cid, resolution=resolution)]}
        self.engine(propose=prop2).run_batch(
            batch("u", "b2", T2, [msg("m2", "画册的事有进展", T2)]))
        return cid, self.store.get_claim("u", cid)

    def test_plan_completed_keeps_text_and_marks_completed(self):
        _cid, claim = self._plan("completed")
        self.assertEqual(claim["status"], "completed")
        self.assertEqual(claim["kind"], "plan")
        self.assertEqual(claim["text"], self.PLAN)

    def test_plan_abandoned(self):
        _cid, claim = self._plan("abandoned")
        self.assertEqual(claim["status"], "abandoned")

    def test_plan_in_progress(self):
        _cid, claim = self._plan("in_progress")
        self.assertEqual(claim["status"], "in_progress")
        self.assertIn("[in_progress][plan]", self.store.memory_path("u").read_text("utf-8"))


class UnknownEdgeUnaffectedTests(Base):
    """The C1 durable-edge behaviour must not regress."""

    def _unknown_correct(self):
        old = self.seed_undecided()
        prop = lambda s, p: {"operations": [
            op("用户表示并非不愿接", "m2", quote="不是不想接", op="correct", target_claim_id=old)]}
        verify = lambda s, p: verdicts(p, "supported", relation="unknown")
        self.engine(propose=prop, verify=verify).run_batch(
            batch("u", "b2", T2, [msg("m2", "不是不想接", T2)]))
        new = [c for c in self.store.list_claims("u") if c["claim_id"] != old][0]
        return old, new

    def test_unknown_correct_still_pending_not_decided(self):
        old, new = self._unknown_correct()
        self.assertEqual(new["status"], "unconfirmed")
        self.assertEqual(new["edge"], "unknown")
        self.assertEqual(self.store.get_claim("u", old)["status"], "contested")

    def test_confirm_of_unknown_edge_still_does_not_close_chain(self):
        old, new = self._unknown_correct()
        prop = lambda s, p: {"operations": [
            op(new["text"], "m3", quote="对，我并非不愿接", op="confirm",
               target_claim_id=new["claim_id"])]}
        self.engine(propose=prop).run_batch(
            batch("u", "b3", T2, [msg("m3", "对，我并非不愿接", T2)]))
        self.assertEqual(self.store.get_claim("u", old)["status"], "contested")
        self.assertNotIn("[superseded]", self.store.memory_path("u").read_text("utf-8"))


class TransactionTests(Base):
    def test_summary_failure_rolls_back_the_decision(self):
        old = self.seed_undecided()
        before_version = self.store.current_version("u")

        def boom(_s, _p):
            raise ModelError("synthetic summarize failure")

        prop = lambda s, p: {"operations": [
            op(DECIDED, "m2", quote="那单我接了", kind="user_fact",
               target_claim_id=old, op="resolve", resolution="completed", importance="high")]}
        r = self.engine(propose=prop, summarize=boom).run_batch(
            batch("u", "b-decide", T2, [msg("m2", "那单我接了", T2)]))
        self.assertEqual(r["status"], "error")
        self.assertNotIn(DECIDED, self.claim_texts())
        self.assertEqual(self.store.get_claim("u", old)["status"], "active")
        self.assertEqual(self.store.get_claim("u", old)["kind"], "undecided")
        self.assertIsNone(self.store.get_claim("u", old)["replaced_by"])
        self.assertEqual(self.store.current_version("u"), before_version)
        self.assertFalse(self.store.has_batch("u", "b-decide"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
