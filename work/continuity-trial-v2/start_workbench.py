#!/usr/bin/env python3
"""Start only the isolated workbench; credentials are explicitly configured for this package."""
from __future__ import annotations
import argparse
from contextlib import closing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from continuity.workbench import serve_workbench
from continuity.config import load_config
from continuity import db


def main(argv=None):
    parser = argparse.ArgumentParser(description='Local memory workbench')
    parser.add_argument('--host', default='127.0.0.1', choices=['127.0.0.1', 'localhost', '::1'])
    parser.add_argument('--port', type=int, default=8794)
    parser.add_argument('--root', type=Path, help='private workbench evaluation root')
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('port must be between 1 and 65535')
    source = load_config()
    source.ensure_dirs()
    # A fresh package has no baseline DB, but session snapshots need its schema.
    # Initialize only a new local DB; the workbench factory keeps inputs read-only.
    if not source.db_path.exists():
        with closing(db.connect(source.db_path)) as conn:
            db.init_db(conn)
    server = serve_workbench(source_config=source, host=args.host, port=args.port, root=args.root)
    print(f'Memory workbench http://{args.host}:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
