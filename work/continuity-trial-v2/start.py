#!/usr/bin/env python3
"""Single entry point for the isolated continuity trial.

    python3 start.py --port 8794

Loopback only. Credentials are supplied explicitly through the environment.
The packaged launchers do not discover or inject credentials from old projects.

This only *starts* the staging server when you run it. It is not started by the
test suite and does not touch the live 8788 service.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from continuity.api import serve  # noqa: E402
from continuity.config import load_config  # noqa: E402
from continuity.maintenance import CliMaintenanceRunner  # noqa: E402
from continuity.queue import MaintenanceQueue  # noqa: E402

LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="start.py", description="Continuity trial staging server")
    ap.add_argument("--port", type=int, default=8794)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--state-dir", default=None, help="override maintenance state dir")
    ap.add_argument("--db", default=None, help="override chat SQLite path")
    ap.add_argument("--no-worker", action="store_true", help="start without the maintenance thread")
    args = ap.parse_args(argv)

    if args.host not in LOOPBACK:
        print(f"refusing non-loopback host: {args.host!r} (loopback only)", file=sys.stderr)
        return 2
    if not (1 <= args.port <= 65535):
        print(f"invalid port: {args.port}", file=sys.stderr)
        return 2

    config = load_config()
    if args.state_dir:
        config.maint_state_dir = Path(args.state_dir)
    if args.db:
        config.db_path = Path(args.db)
    config.ensure_dirs()

    queue = MaintenanceQueue(config, CliMaintenanceRunner(config))
    if not args.no_worker:
        queue.start()
    server = serve(config, queue, host=args.host, port=args.port)
    print(
        f"continuity trial staging on http://{args.host}:{args.port}\n"
        f"  chat db   : {config.db_path}\n"
        f"  maint state: {config.maint_state_dir}\n"
        f"  main=Pro {config.main_chat_model} · retrieval=Flash {config.chat_model}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        queue.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
