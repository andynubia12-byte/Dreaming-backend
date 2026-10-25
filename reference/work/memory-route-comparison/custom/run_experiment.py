"""Custom (self-written) route experiment over the FROZEN shared benchmark.

Uses the REAL pipeline of work/letta-prototype unchanged:
    MemoryWorker -> agents.verify_source_ids -> agents.memory_agent_consolidate
    -> ideastore.save/version   (background memory worker + memory agent + ideastore)

Isolation / documented adaptations (differences from the original runtime):
  1. Isolated data: db is a fresh `sqlite .backup` copy; working memory lives under
     custom/run/. The original prototype's DB, memory, and 8787 service are untouched.
  2. Historical `now`: the original worker passes `now_shanghai()` (real clock). Here
     we monkeypatch `lettaproto.worker.now_shanghai` per batch to the batch end from
     the benchmark, so no future data leaks in and section dates reflect the stage.
  3. One queue task per batch (kind="suggestion") carrying that batch's source ids.
  4. Budget batching in eval: one native retrieval call per (checkpoint, theme) and
     one answer call per checkpoint (instead of per question), to stay inside the
     ~80-request engineering checkpoint. Recorded in the report.

Nothing is sent anywhere except api.deepseek.com (deepseek-flash).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                 # memory-route-comparison
PROTO = ROOT.parent / "letta-prototype"                    # read-only source
BENCH = ROOT / "shared" / "benchmark.json"
SRC_DB = PROTO / "db" / "canonical.sqlite"
RUN = Path(__file__).resolve().parent / "run"
SH = dt.timezone(dt.timedelta(hours=8), "Asia/Shanghai")

sys.path.insert(0, str(PROTO / "src"))


def log(msg: str) -> None:
    print(f"[{dt.datetime.now(SH).strftime('%H:%M:%S')}] {msg}", flush=True)


def make_config(usage_file: Path):
    from lettaproto.config import Config

    class Cfg(Config):
        def ensure_dirs(self) -> None:
            for p in (self.db_path.parent, self.exports_dir, self.runs_dir, self.memory_root,
                      self.memory_root / INSTANCE / "versions"):
                p.mkdir(parents=True, exist_ok=True)

    cfg = Cfg()
    cfg.db_path = RUN / "canonical.sqlite"
    cfg.exports_dir = RUN / "exports"
    cfg.runs_dir = RUN / "runs"
    cfg.memory_root = RUN / "memory"
    cfg.usage_file = usage_file
    cfg.chat_model = "deepseek-flash"
    cfg.api_base_url = "https://api.deepseek.com"
    cfg.max_requests = 400
    cfg.trial_max_requests = 400
    cfg.max_input_tokens = 20_000_000
    cfg.updater_max_input_tokens = 200_000
    cfg.ensure_dirs()
    return cfg


INSTANCE = "run"
# ONE generic fix (kept in custom/ only): deepseek-flash spends most of the
# completion budget on reasoning tokens, so the original ceilings (memory agent
# max_tokens=4000; retrieval 2200/3000) truncate the answer (finish=length, empty
# content). We raise the completion ceiling for every model call in this harness.
# This does NOT change the prompt, the evidence, or any other pipeline logic.
FIX_MIN_COMPLETION = 16000


def instrument_client(config):
    """DeepSeekClient subclass that records one JSON line per API call."""
    from lettaproto import deepseek

    calls_path = RUN / "calls.jsonl"

    class LoggedClient(deepseek.DeepSeekClient):
        def chat(self, *a, **kw):
            kw["max_tokens"] = max(int(kw.get("max_tokens") or 0), FIX_MIN_COMPLETION)
            t0 = time.time()
            rec = {"ts": dt.datetime.now(SH).isoformat(), "model": kw.get("model") or self.config.chat_model,
                   "max_tokens": kw.get("max_tokens"), "tag": kw.pop("tag", None)}
            try:
                payload = super().chat(*a, **kw)
            except Exception as exc:  # noqa: BLE001
                rec.update({"ok": False, "latency_s": round(time.time() - t0, 2),
                            "error": f"{type(exc).__name__}: {exc}"[:300]})
                _append(calls_path, rec)
                raise
            u = payload.get("usage") or {}
            rec.update({"ok": True, "latency_s": round(time.time() - t0, 2),
                        "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                        "finish": (payload.get("choices") or [{}])[0].get("finish_reason")})
            _append(calls_path, rec)
            return payload

    return LoggedClient


def _append(path: Path, obj: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def setup_db() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    if not (RUN / "canonical.sqlite").exists():
        src = sqlite3.connect(f"file:{SRC_DB}?mode=ro", uri=True)
        dst = sqlite3.connect(str(RUN / "canonical.sqlite"))
        with dst:
            src.backup(dst)
        src.close()
        dst.close()
        log("snapshot: copied canonical.sqlite via sqlite .backup into run/")


def consolidate(cfg, bench, cold: bool) -> None:
    from lettaproto import agents, db, ideastore, worker as worker_mod

    if cold:
        conn = db.connect(cfg.db_path)
        db.init_db(conn)
        row = conn.execute("SELECT COUNT(*) c FROM idea_versions WHERE instance=?", (INSTANCE,)).fetchone()
        conn.close()
        if row["c"]:
            raise SystemExit(f"instance '{INSTANCE}' already has versions; refuse cold restart (use --resume)")
        conn = db.connect(cfg.db_path)
        ideastore.save(cfg, conn, "# 长期记忆\n", note="cold start (empty memory)", instance=INSTANCE)
        conn.close()
        log("seeded empty working memory for instance 'run'")

    w = worker_mod.MemoryWorker(cfg)
    Client = instrument_client(cfg)
    # Route the worker's memory agent through the logging client.
    orig_consolidate = agents.memory_agent_consolidate

    def logged_consolidate(config, conn, **kw):
        kw.setdefault("client", Client(config))
        return orig_consolidate(config, conn, **kw)

    for b in bench["batches"]:
        bid, now_iso = b["batch_id"], b["now_iso"]
        sids = [sid for mid in b["message_ids"]
                for sid in next(m for m in bench["messages"] if m["id"] == mid)["source_ids"]]
        worker_mod.now_shanghai = lambda *_a, _n=now_iso, **_k: _n   # historical now
        agents.memory_agent_consolidate = logged_consolidate
        t0 = time.time()
        res = w.enqueue(INSTANCE, "suggestion", {"statement": None}, sids, dedup_key=f"bench:{bid}:fixed")
        tid = res["task_id"]
        out = None
        for _ in range(4):                      # bounded retries (mirrors worker_max_attempts)
            out = w.run_once(INSTANCE)
            if out is None or out.get("status") in ("saved", "failed"):
                break
        log(f"{bid} now={now_iso} task={tid} status={out and out.get('status')} "
            f"({round(time.time()-t0,1)}s, {len(sids)} source ids)")
        cp = RUN / "checkpoints"
        cp.mkdir(exist_ok=True)
        conn = db.connect(cfg.db_path)
        conn.execute("PRAGMA journal_mode=DELETE")
        text = ideastore.load(cfg, INSTANCE)
        ver = ideastore.current_version(conn, INSTANCE)
        (cp / f"{bid}_idea.md").write_text(text, "utf-8")
        conn.close()
        log(f"  checkpoint {bid}: version={ver} chars={len(text)}")
    agents.memory_agent_consolidate = orig_consolidate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["consolidate", "eval", "all"], default="all")
    ap.add_argument("--cold", action="store_true", help="wipe and restart the working memory")
    args = ap.parse_args()

    bench = json.loads(BENCH.read_text("utf-8"))
    setup_db()
    cfg = make_config(RUN / "usage.json")
    if not cfg.api_key:
        raise SystemExit("DEEPSEEK_API_KEY missing; run through the credential wrapper")
    if args.cold:
        # archive the pre-fix run as evidence (before/after comparison)
        bf = RUN / "before_fix"
        bf.mkdir(exist_ok=True)
        if (RUN / "calls.jsonl").exists():
            shutil.copy(RUN / "calls.jsonl", bf / "calls_before.jsonl")
        if (RUN / "canonical.sqlite").exists():
            c = sqlite3.connect(str(RUN / "canonical.sqlite"))
            rows = c.execute("SELECT id,instance,kind,status,attempts,error FROM memory_tasks ORDER BY id").fetchall()
            c.close()
            (bf / "tasks_before.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf-8")
        if (RUN / "checkpoints").exists():
            for f in (RUN / "checkpoints").glob("b*_idea.md"):
                shutil.copy(f, bf / ("before_" + f.name))
        for p in (RUN / "memory", RUN / "checkpoints"):
            if p.exists():
                shutil.rmtree(p)
        if (RUN / "eval").exists():
            shutil.rmtree(RUN / "eval")
        for f in (RUN / "usage.json", RUN / "calls.jsonl"):
            if f.exists():
                f.unlink()
        # drop this instance's rows from the copied DB so the cold start is truly clean
        c = sqlite3.connect(str(RUN / "canonical.sqlite"))
        with c:
            for tbl in ("idea_versions", "memory_updates", "memory_tasks", "search_traces", "chat_log"):
                c.execute(f"DELETE FROM {tbl} WHERE instance=?", (INSTANCE,))
        c.close()
    if args.phase in ("consolidate", "all"):
        consolidate(cfg, bench, cold=args.cold)
    if args.phase in ("eval", "all"):
        import run_eval
        run_eval.main(cfg, bench)


if __name__ == "__main__":
    main()
