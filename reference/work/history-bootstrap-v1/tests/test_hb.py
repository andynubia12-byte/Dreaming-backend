"""Offline regressions for history-bootstrap-v1 (no network, no demo, no real export)."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import bootstrap  # noqa: E402
import hb_core  # noqa: E402

FIX = ROOT / "fixtures"


def _digest_tree(folder: Path) -> dict:
    out = {}
    for p in sorted(folder.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(folder))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def build(self, fixture: str, out_name="out", as_of="2027-01-01T00:00:00+08:00", window="month"):
        out = self.tmp / out_name
        argv = ["build", "--export-dir", str(FIX / fixture), "--out", str(out),
                "--user-id", "u", "--window", window]
        if as_of:
            argv += ["--as-of", as_of]
        rc = bootstrap.main(argv)
        return rc, out


class BranchAndRoleTests(Base):
    def test_selected_branch_only_and_roles(self):
        _, out = self.build("tree_min")
        recs = [json.loads(l) for l in (out / "evidence.jsonl").read_text("utf-8").splitlines()]
        nodes = {r["node_id"] for r in recs}
        self.assertEqual(nodes, {"n1", "n2b", "n4", "n6", "nU"})     # selected path only
        self.assertNotIn("n2a", nodes)                               # other branch excluded
        self.assertNotIn("orphan", nodes)                            # broken-parent node excluded
        self.assertNotIn("cyc1", nodes)                              # cycle node excluded
        roles = {r["role"] for r in recs}
        self.assertEqual(roles, {"user", "assistant"})
        self.assertFalse(any(r["content_type"] in ("thoughts", "reasoning_recap") for r in recs))

    def test_counts(self):
        rc, out = self.build("tree_min")
        self.assertEqual(rc, 0)
        man = json.loads((out / "manifest.json").read_text("utf-8"))
        c = man["counts"]
        self.assertEqual(c["conversations"], 2)
        self.assertEqual(c["nodes_total"], 12)
        self.assertEqual(c["nodes_selected"], 7)
        self.assertEqual(c["evidence_messages"], 5)
        self.assertEqual(c["user_messages"], 2)
        self.assertEqual(c["assistant_messages"], 3)
        self.assertEqual(c["skipped_reasoning"], 1)
        self.assertEqual(c["skipped_system"], 1)
        self.assertEqual(c["attachments_gap"], 1)

    def test_missing_current_node_reported(self):
        _, out = self.build("tree_min")
        man = json.loads((out / "manifest.json").read_text("utf-8"))
        miss = man["branch_issues"]["missing_current_node"]
        self.assertEqual([m["conversation_id"] for m in miss], ["convB"])


class TimeTests(Base):
    def test_message_time_is_message_create_time_not_update_time(self):
        _, out = self.build("tree_min")
        recs = [json.loads(l) for l in (out / "evidence.jsonl").read_text("utf-8").splitlines()]
        # convA.update_time == 1999999999.0 (~2033) must never be used as a message time
        self.assertTrue(all(r["message_at"] is None or r["message_at"].startswith("2026-") for r in recs))
        n1 = next(r for r in recs if r["node_id"] == "n1")
        self.assertEqual(n1["message_at"], "2026-01-05T12:00:00+08:00")
        self.assertEqual(n1["time_source"], "message.create_time")

    def test_out_of_order_recorded_not_dropped(self):
        _, out = self.build("tree_min")
        man = json.loads((out / "manifest.json").read_text("utf-8"))
        self.assertEqual(man["counts"]["time_out_of_order"], 1)
        self.assertEqual(man["out_of_order"][0]["node_id"], "n4")
        recs = [json.loads(l) for l in (out / "evidence.jsonl").read_text("utf-8").splitlines()]
        self.assertIn("n4", {r["node_id"] for r in recs})          # kept, not silently dropped

    def test_undated_kept_in_evidence_but_not_batched(self):
        _, out = self.build("tree_min")
        recs = [json.loads(l) for l in (out / "evidence.jsonl").read_text("utf-8").splitlines()]
        nU = next(r for r in recs if r["node_id"] == "nU")
        self.assertTrue(nU["time_missing"])
        self.assertIsNone(nU["message_at"])
        for bfile in (out / "batches").glob("*.json"):
            ids = {m["id"] for m in json.loads(bfile.read_text("utf-8"))["messages"]}
            self.assertNotIn("mAU", ids)                            # never guess a time

    def test_attachment_is_gap_only(self):
        _, out = self.build("tree_min")
        recs = [json.loads(l) for l in (out / "evidence.jsonl").read_text("utf-8").splitlines()]
        n4 = next(r for r in recs if r["node_id"] == "n4")
        self.assertTrue(n4["attachments"])
        self.assertTrue(all(a["content"] is None and a["gap"] for a in n4["attachments"]))
        self.assertEqual(n4["attachments"][0]["name"], "sleep.png")


class BatchTests(Base):
    def test_batches_fixed_time_boundary_no_future_evidence(self):
        _, out = self.build("tree_min")
        files = sorted((out / "batches").glob("*.json"))
        names = [f.name for f in files]
        self.assertEqual(names, ["batch-000001.json", "batch-000002.json"])   # zero-padded prefix
        self.assertEqual(names, sorted(names))                                # lexicographic == chronological
        windows = []
        for f in files:
            b = json.loads(f.read_text("utf-8"))
            self.assertEqual(set(b) >= {"user_id", "batch_id", "now", "messages"}, True)
            windows.append(b["_meta"]["window"])
            for m in b["messages"]:
                self.assertLess(m["message_at"], b["now"])                    # never future evidence
        self.assertEqual(windows, ["2026-01", "2026-02"])
        self.assertEqual(len(json.loads(files[0].read_text("utf-8"))["messages"]), 3)

    def test_batch_json_is_cli_compatible_shape(self):
        _, out = self.build("tree_min")
        b = json.loads((out / "batches" / "batch-000001.json").read_text("utf-8"))
        for key in ("user_id", "batch_id", "now", "messages"):
            self.assertIn(key, b)
        for m in b["messages"]:
            self.assertEqual(set(m) >= {"id", "conversation_id", "role", "message_at", "text"}, True)
            self.assertIn(m["role"], ("user", "assistant"))


class DuplicateTests(Base):
    def test_duplicate_same_content_is_deduped(self):
        rc, out = self.build("tree_dup_ok")
        self.assertEqual(rc, 0)
        man = json.loads((out / "manifest.json").read_text("utf-8"))
        self.assertEqual(man["counts"]["evidence_messages"], 1)
        self.assertEqual(man["counts"]["deduped"], 1)

    def test_duplicate_conflict_fails_explicitly(self):
        rc, out = self.build("tree_dup_conflict")
        self.assertEqual(rc, 2)
        man = json.loads((out / "manifest.json").read_text("utf-8"))
        self.assertEqual(man["status"], "failed")
        self.assertEqual(len(man["duplicate_conflicts"]), 1)
        self.assertFalse((out / "evidence.jsonl").exists())           # fail closed
        self.assertFalse((out / "batches").exists())


class BranchIssueTests(Base):
    def test_cycle_reported(self):
        rc, out = self.build("tree_cycle")
        self.assertEqual(rc, 0)
        man = json.loads((out / "manifest.json").read_text("utf-8"))
        self.assertTrue(man["branch_issues"]["cycle"])

    def test_broken_parent_reported(self):
        rc, out = self.build("tree_broken")
        self.assertEqual(rc, 0)
        man = json.loads((out / "manifest.json").read_text("utf-8"))
        self.assertEqual(man["branch_issues"]["broken_parent"][0]["parent"], "ghost")


class MultimodalTests(Base):
    def test_transcript_dict_text_is_preserved(self):
        _, out = self.build("tree_multimodal")
        recs = [json.loads(l) for l in (out / "evidence.jsonl").read_text("utf-8").splitlines()]
        u1 = next(r for r in recs if r["node_id"] == "u1")
        self.assertIn("这是语音转写的用户口述内容。", u1["text"])       # dict.text kept
        self.assertNotIn("{", u1["text"])                                # never str(dict)
        self.assertNotIn("asset_pointer", u1["text"])
        self.assertEqual(u1["content_type"], "multimodal_text")

    def test_media_parts_are_gaps_not_content(self):
        _, out = self.build("tree_multimodal")
        recs = [json.loads(l) for l in (out / "evidence.jsonl").read_text("utf-8").splitlines()]
        u1 = next(r for r in recs if r["node_id"] == "u1")
        self.assertGreaterEqual(u1["omitted_non_text_parts"], 1)         # omitted count explicit
        self.assertTrue(u1["attachments"])
        self.assertTrue(all(a["content"] is None and a["gap"] for a in u1["attachments"]))
        mimes = {a["mime_type"] for a in u1["attachments"]}
        self.assertIn("image_asset_pointer", mimes)
        self.assertIn("audio", mimes)

    def test_manifest_records_text_policy(self):
        _, out = self.build("tree_multimodal")
        man = json.loads((out / "manifest.json").read_text("utf-8"))
        self.assertIn("dict.text parts", man["text_policy"])
        self.assertIn("no outer strip", man["text_policy"])


class PlanAndSafetyTests(Base):
    def test_plan_exposes_growth_and_does_not_write(self):
        plan = hb_core.build_plan(FIX / "tree_min", user_id="u")
        self.assertEqual(plan["status"], "ok")
        self.assertEqual(plan["batch_count"], 2)
        prev = 0
        for b in plan["batches"]:
            self.assertGreaterEqual(b["est_input_tokens_coarse"], prev)   # coarse, monotonic
            prev = b["est_input_tokens_coarse"]
            self.assertEqual(b["cumulative_evidence_chars_after"],
                             b["prior_evidence_chars"] + b["new_chars"])
            self.assertNotIn("est_input_tokens_upper", b)                 # no false "upper bound"
        self.assertEqual(plan["batches"][0]["prior_evidence_chars"], 0)
        self.assertGreater(plan["batches"][1]["prior_evidence_chars"], 0)     # growth exposed
        self.assertIn("NOT a token upper bound", " ".join(plan["notes"]))
        self.assertIn("full-scale adaptation is NOT demonstrated", " ".join(plan["notes"]))

    def test_input_is_read_only(self):
        before = _digest_tree(FIX / "tree_min")
        self.build("tree_min")
        after = _digest_tree(FIX / "tree_min")
        self.assertEqual(before, after)

    def test_out_inside_export_refused(self):
        with self.assertRaises(SystemExit):
            bootstrap.main(["build", "--export-dir", str(FIX / "tree_min"),
                            "--out", str(FIX / "tree_min" / "inside"),
                            "--as-of", "2027-01-01T00:00:00+08:00"])

    def test_retrieval_import_schema(self):
        import sqlite3
        _, out = self.build("tree_min")
        conn = sqlite3.connect(str(out / "retrieval_import.sqlite"))
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(messages)")}
            self.assertTrue({"conversation_id", "node_id", "message_id", "role", "create_time",
                             "text", "is_selected_branch", "is_assistant_reasoning"} <= cols)
            # the demo's TrialRetrieval query shape works against this DB
            rows = conn.execute(
                "SELECT conversation_id, node_id, message_id, role, create_time, text FROM messages "
                "WHERE is_selected_branch=1 AND is_assistant_reasoning=0 "
                "AND text IS NOT NULL AND LENGTH(text)>0 AND create_time < ? ORDER BY create_time, node_id",
                (hb_core.dt.datetime(2027, 1, 1, tzinfo=hb_core.SH).timestamp(),)).fetchall()
            self.assertEqual(len(rows), 4)   # nU has no create_time -> excluded by the time guard
        finally:
            conn.close()


class AsOfTests(Base):
    def test_build_requires_as_of(self):
        with self.assertRaises(SystemExit):
            bootstrap.main(["build", "--export-dir", str(FIX / "tree_min"), "--out", str(self.tmp / "x")])

    def test_as_of_must_be_tz_aware(self):
        with self.assertRaises(SystemExit):
            bootstrap.main(["build", "--export-dir", str(FIX / "tree_min"),
                            "--out", str(self.tmp / "x"), "--as-of", "2026-02-01T00:00:00"])

    def test_as_of_exactly_boundary(self):
        rc, out = self.build("tree_min", as_of="2026-02-01T00:00:00+08:00")
        self.assertEqual(rc, 0)
        man = json.loads((out / "manifest.json").read_text("utf-8"))
        self.assertEqual(man["as_of"], "2026-02-01T00:00:00+08:00")
        files = sorted((out / "batches").glob("*.json"))
        self.assertEqual([f.name for f in files], ["batch-000001.json"])   # future month excluded
        b = json.loads(files[0].read_text("utf-8"))
        self.assertEqual(b["now"], "2026-02-01T00:00:00+08:00")            # now == as_of boundary
        self.assertEqual({m["id"] for m in b["messages"]}, {"mA1", "mA4", "mA2b"})
        self.assertEqual(man["counts"]["excluded_after_as_of"], 1)          # mA6 (Feb 3)
        for f in files:
            bb = json.loads(f.read_text("utf-8"))
            self.assertLessEqual(bb["now"], man["as_of"])
            for m in bb["messages"]:
                self.assertLess(m["message_at"], bb["now"])

    def test_as_of_mid_period_clamps_last_batch(self):
        rc, out = self.build("tree_min", as_of="2026-02-15T00:00:00+08:00")
        files = sorted((out / "batches").glob("*.json"))
        self.assertEqual(len(files), 2)
        b2 = json.loads(files[1].read_text("utf-8"))
        self.assertEqual(b2["now"], "2026-02-15T00:00:00+08:00")            # min(boundary, as_of)
        self.assertTrue(b2["_meta"]["clamped_by_as_of"])
        self.assertEqual({m["id"] for m in b2["messages"]}, {"mA6"})

    def test_as_of_excludes_later_messages(self):
        rc, out = self.build("tree_min", as_of="2026-01-15T00:00:00+08:00")
        ids = {json.loads(l)["message_id"] for l in (out / "evidence.jsonl").read_text("utf-8").splitlines()}
        self.assertIn("mA1", ids)
        self.assertIn("mA4", ids)
        self.assertNotIn("mA2b", ids)      # Jan 20 excluded
        self.assertNotIn("mA6", ids)       # Feb excluded
        self.assertIn("mAU", ids)          # undated retained (flagged), never batched
        man = json.loads((out / "manifest.json").read_text("utf-8"))
        self.assertEqual(man["counts"]["excluded_after_as_of"], 2)
        b = json.loads(sorted((out / "batches").glob("*.json"))[0].read_text("utf-8"))
        self.assertEqual({m["id"] for m in b["messages"]}, {"mA1", "mA4"})
        self.assertEqual(b["now"], "2026-01-15T00:00:00+08:00")


class DayWindowTests(Base):
    def test_day_window_batches(self):
        rc, out = self.build("tree_min", window="day", as_of="2027-01-01T00:00:00+08:00")
        names = [f.name for f in sorted((out / "batches").glob("*.json"))]
        self.assertEqual(names, [f"batch-{i:06d}.json" for i in range(1, 5)])   # Jan5/10/20/Feb3
        for f in sorted((out / "batches").glob("*.json")):
            b = json.loads(f.read_text("utf-8"))
            for m in b["messages"]:
                self.assertLess(m["message_at"], b["now"])

    def test_oversize_reported_not_split(self):
        old_m, old_c = hb_core.OVERSIZE_MESSAGES, hb_core.OVERSIZE_CHARS
        hb_core.OVERSIZE_MESSAGES, hb_core.OVERSIZE_CHARS = 1, 10 ** 12
        try:
            plan = hb_core.build_plan(FIX / "tree_min", user_id="u", window="day")
            self.assertTrue(plan["oversized_batches"])                        # reported
            self.assertEqual(plan["batch_count"], len(plan["batches"]))       # not split/truncated
            self.assertEqual(set(plan["oversized_batches"]),
                             {b["batch_id"] for b in plan["batches"] if b["oversize"]})
        finally:
            hb_core.OVERSIZE_MESSAGES, hb_core.OVERSIZE_CHARS = old_m, old_c


if __name__ == "__main__":
    unittest.main(verbosity=2)
