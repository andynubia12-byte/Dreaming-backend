"""Local run entry: idea working memory, query, chat, incremental update, API."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import db, ideastore, importer, vendor
from .api import serve
from .chat import chat_agentic, chat_direct, chat_letta
from .config import load_config
from .context import build_context
from .deepseek import DeepSeekClient
from .letta import compiled_system_prompt, create_agent, memory_dir, write_agent_memory
from .query import query as run_query
from .updater import update_from_correction, update_from_new_messages
from .util import read_json, write_json
from .worker import MemoryWorker


def _agent_file(config):
    return config.runs_dir / "agent.json"


def _agent_id(config, args):
    return getattr(args, "agent", None) or (read_json(_agent_file(config), {}) or {}).get("agent_id")


def _conn(config):
    conn = db.connect(config.db_path)
    db.init_db(conn)
    return conn


def cmd_manifest(config, args):
    files = importer.scan_source(config.export_dir)
    out = config.exports_dir / "source_manifest.json"
    write_json(out, {"export_dir": str(config.export_dir), "files": files})
    print(json.dumps({"files": len(files), "out": str(out)}, ensure_ascii=False))


def cmd_import(config, args):
    report = importer.run_import(config.export_dir, config.db_path, config.exports_dir / "import_report.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def cmd_idea_init(config, args):
    conn = _conn(config)
    out = {"prod": ideastore.seed_prod(config, conn)}
    if args.with_test:
        out["test"] = ideastore.seed_test(config, conn)
    print(json.dumps(out, ensure_ascii=False, indent=2))


def cmd_idea_show(config, args):
    conn = _conn(config)
    info = ideastore.stats(config, conn, args.instance)
    if args.head:
        info["head"] = ideastore.load(config, args.instance)[: args.head]
    print(json.dumps(info, ensure_ascii=False, indent=2))


def cmd_idea_versions(config, args):
    conn = _conn(config)
    print(json.dumps(ideastore.list_versions(conn, args.instance), ensure_ascii=False, indent=2))


def cmd_idea_diff(config, args):
    conn = _conn(config)
    versions = ideastore.list_versions(conn, args.instance)
    if len(versions) < 2:
        print(json.dumps({"error": "need >=2 versions to diff", "count": len(versions)}, ensure_ascii=False))
        return
    old = Path(versions[-2]["path"]).read_text("utf-8")
    new = Path(versions[-1]["path"]).read_text("utf-8")
    print(ideastore.unified_diff(old, new, context=args.context))


def cmd_idea_rollback(config, args):
    conn = _conn(config)
    print(json.dumps(ideastore.rollback(config, conn, args.instance, args.version), ensure_ascii=False, indent=2))


def cmd_query(config, args):
    conn = _conn(config)
    print(json.dumps(run_query(
        config, conn, args.question, instance=args.instance,
        top_k=args.top_k, history_top_k=args.history_top_k, include_history=args.history,
    ), ensure_ascii=False, indent=2))


def cmd_context(config, args):
    conn = _conn(config)
    bundle = build_context(
        config, conn, instance=args.instance,
        recent_limit=args.recent, recent_token_budget=args.recent_tokens,
        from_chatlog=not args.no_from_chatlog,
    )
    write_json(config.exports_dir / f"context_report_{args.instance}.json", bundle.stats)
    (config.exports_dir / f"context_system_prompt_{args.instance}.txt").write_text(bundle.system_prompt(), "utf-8")
    print(json.dumps(bundle.stats, ensure_ascii=False, indent=2))


def cmd_chat(config, args):
    conn = _conn(config)
    if getattr(args, "agentic", False):
        result = chat_agentic(
            config, conn, args.prompt, instance=args.instance,
            recent_limit=args.recent, recent_token_budget=args.recent_tokens,
            from_chatlog=not args.no_from_chatlog, worker=MemoryWorker(config),
        )
    else:
        result = chat_direct(
            config, conn, args.prompt, instance=args.instance,
            recent_limit=args.recent, recent_token_budget=args.recent_tokens,
            from_chatlog=not args.no_from_chatlog,
        )
    if args.update:
        client = DeepSeekClient(config)
        result["update"] = update_from_new_messages(config, conn, client, args.instance)
        result["api_usage_total"] = client.usage.to_dict()
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_update(config, args):
    conn = _conn(config)
    client = DeepSeekClient(config)
    result = update_from_new_messages(config, conn, client, args.instance, apply=not args.dry_run)
    result["api_usage_total"] = client.usage.to_dict()
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_correct(config, args):
    conn = _conn(config)
    client = DeepSeekClient(config)
    result = update_from_correction(config, conn, client, args.text, args.instance, apply=not args.dry_run)
    result["api_usage_total"] = client.usage.to_dict()
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_vendor_letta(config, args):
    meta = vendor.vendor_letta(config, force=args.force)
    print(json.dumps({"meta": meta, "verify": vendor.verify(config), "letta_entry": str(config.letta_entry)},
                     ensure_ascii=False, indent=2))


def cmd_letta_init(config, args):
    conn = _conn(config)
    data = create_agent(config)
    agent_id = data.get("agent_id")
    if agent_id:
        write_json(_agent_file(config), {"agent_id": agent_id, "model": config.letta_model})
        data["memory_sync"] = write_agent_memory(
            config, agent_id, ideastore.load(config, args.instance), commit_message="init: sync idea.md"
        )
    print(json.dumps(data, ensure_ascii=False, indent=2)[:1500])


def cmd_letta_memory(config, args):
    agent_id = _agent_id(config, args)
    info = write_agent_memory(config, agent_id, ideastore.load(config, args.instance), commit_message=args.message)
    info["memory_dir"] = str(memory_dir(config, agent_id))
    print(json.dumps(info, ensure_ascii=False, indent=2))


def cmd_chat_letta(config, args):
    conn = _conn(config)
    print(json.dumps(chat_letta(config, conn, args.prompt, _agent_id(config, args), instance=args.instance),
                     ensure_ascii=False, indent=2)[:2500])


def cmd_compiled(config, args):
    data = compiled_system_prompt(config, _agent_id(config, args))
    if not data:
        print(json.dumps({"error": "no compiled system prompt found"}, ensure_ascii=False))
        return
    content = data["data"].get("content") or ""
    print(json.dumps({
        "path": data["path"],
        "content_chars": data["content_chars"],
        "coreMemory_chars": len(data["data"].get("coreMemory") or ""),
        "contains": {k: (k in content) for k in args.contains},
    }, ensure_ascii=False, indent=2))


def cmd_api(config, args):
    server = serve(config, host="127.0.0.1", port=args.port)
    print(json.dumps({"listening": f"http://127.0.0.1:{args.port}", "pid": os.getpid()}, ensure_ascii=False), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def cmd_worker(config, args):
    w = MemoryWorker(config)
    if args.once:
        recovered = w.recover()
        result = w.run_once(args.instance)
        print(json.dumps({"recovered": recovered, "result": result, "counts": w.counts(args.instance)},
                         ensure_ascii=False, indent=2))
        return
    import threading

    w.start()
    print(json.dumps({"worker": "running", "poll_seconds": config.worker_poll_seconds}, ensure_ascii=False), flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        w.stop()


def cmd_tasks(config, args):
    w = MemoryWorker(config)
    payload = {"counts": w.counts(args.instance), "tasks": w.list_tasks(args.instance, limit=args.limit)}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("counts:", json.dumps(payload["counts"], ensure_ascii=False))
        for t in payload["tasks"]:
            print(f"#{t['id']} {t['kind']:10s} {t['status']:10s} src={t['source_ids']} err={t['error'] or ''}")


def cmd_enqueue_update(config, args):
    w = MemoryWorker(config)
    spec = {"statement": args.statement} if args.statement else {"trigger": "manual"}
    result = w.enqueue(args.instance, args.kind, spec, args.source or [])
    print(json.dumps({**result, "counts": w.counts(args.instance)}, ensure_ascii=False, indent=2))


def cmd_budget(config, args):
    if args.reset:
        DeepSeekClient(config).reset_budget()
    print(json.dumps(read_json(config.usage_file, {}), ensure_ascii=False, indent=2))


def cmd_status(config, args):
    conn = _conn(config)
    imported = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] > 0
    payload = {
        "export_dir": str(config.export_dir),
        "db": str(config.db_path),
        "agent": read_json(_agent_file(config), {}) or {},
        "letta_home": str(config.letta_home),
        "letta_entry": str(config.letta_entry),
        "letta_vendored": vendor.is_vendored(config),
        "api_usage": read_json(config.usage_file, {}),
        "imported": imported,
        "idea": {"prod": ideastore.stats(config, conn, "prod"), "test": ideastore.stats(config, conn, "test")},
        "import_stats": importer.stats(conn) if imported else {},
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lettaproto", description="Letta real-data prototype (idea.md centered)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifest").set_defaults(func=cmd_manifest)
    sub.add_parser("import").set_defaults(func=cmd_import)

    ii = sub.add_parser("idea-init"); ii.add_argument("--with-test", action="store_true"); ii.set_defaults(func=cmd_idea_init)
    ish = sub.add_parser("idea-show"); ish.add_argument("--instance", default="prod"); ish.add_argument("--head", type=int, default=0); ish.set_defaults(func=cmd_idea_show)
    iv = sub.add_parser("idea-versions"); iv.add_argument("--instance", default="prod"); iv.set_defaults(func=cmd_idea_versions)
    idf = sub.add_parser("idea-diff"); idf.add_argument("--instance", default="prod"); idf.add_argument("--context", type=int, default=2); idf.set_defaults(func=cmd_idea_diff)
    ir = sub.add_parser("idea-rollback"); ir.add_argument("--instance", default="prod"); ir.add_argument("--version", type=int, required=True); ir.set_defaults(func=cmd_idea_rollback)

    q = sub.add_parser("query")
    q.add_argument("--question", required=True); q.add_argument("--instance", default="prod"); q.add_argument("--history", action="store_true")
    q.add_argument("--top-k", type=int, default=5); q.add_argument("--history-top-k", type=int, default=8); q.set_defaults(func=cmd_query)

    c = sub.add_parser("context"); c.add_argument("--instance", default="prod"); c.add_argument("--recent", type=int, default=None); c.add_argument("--recent-tokens", type=int, default=None); c.add_argument("--no-from-chatlog", action="store_true"); c.set_defaults(func=cmd_context)

    ch = sub.add_parser("chat")
    ch.add_argument("--prompt", required=True); ch.add_argument("--instance", default="prod"); ch.add_argument("--recent", type=int, default=None)
    ch.add_argument("--recent-tokens", type=int, default=None); ch.add_argument("--no-from-chatlog", action="store_true")
    ch.add_argument("--agentic", action="store_true", help="main model tool loop (search_memory / submit_memory_update)")
    ch.add_argument("--update", action="store_true", help="explicitly run the incremental updater after replying (off by default)")
    ch.set_defaults(func=cmd_chat)

    vl = sub.add_parser("vendor-letta"); vl.add_argument("--force", action="store_true"); vl.set_defaults(func=cmd_vendor_letta)

    wk = sub.add_parser("worker"); wk.add_argument("--once", action="store_true"); wk.add_argument("--instance", default=None); wk.set_defaults(func=cmd_worker)
    tk = sub.add_parser("tasks"); tk.add_argument("--instance", default="prod"); tk.add_argument("--limit", type=int, default=20); tk.add_argument("--json", action="store_true"); tk.set_defaults(func=cmd_tasks)
    eu = sub.add_parser("enqueue-update"); eu.add_argument("--instance", default="prod"); eu.add_argument("--kind", default="idle"); eu.add_argument("--statement"); eu.add_argument("--source", nargs="*"); eu.set_defaults(func=cmd_enqueue_update)

    up = sub.add_parser("update"); up.add_argument("--instance", default="prod"); up.add_argument("--dry-run", action="store_true"); up.set_defaults(func=cmd_update)
    co = sub.add_parser("correct"); co.add_argument("--text", required=True); co.add_argument("--instance", default="prod"); co.add_argument("--dry-run", action="store_true"); co.set_defaults(func=cmd_correct)

    li = sub.add_parser("letta-init"); li.add_argument("--instance", default="prod"); li.set_defaults(func=cmd_letta_init)
    lm = sub.add_parser("letta-memory"); lm.add_argument("--instance", default="prod"); lm.add_argument("--agent"); lm.add_argument("--message", default="sync: idea.md -> agent memory"); lm.set_defaults(func=cmd_letta_memory)
    cl = sub.add_parser("chat-letta"); cl.add_argument("--prompt", required=True); cl.add_argument("--instance", default="prod"); cl.add_argument("--agent"); cl.set_defaults(func=cmd_chat_letta)
    cp = sub.add_parser("compiled"); cp.add_argument("--agent"); cp.add_argument("--contains", nargs="*", default=[]); cp.set_defaults(func=cmd_compiled)
    ap = sub.add_parser("api"); ap.add_argument("--port", type=int, default=8787); ap.set_defaults(func=cmd_api)
    bg = sub.add_parser("budget"); bg.add_argument("--reset", action="store_true"); bg.set_defaults(func=cmd_budget)
    sub.add_parser("status").set_defaults(func=cmd_status)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    args.func(load_config(), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
