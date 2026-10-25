#!/usr/bin/env python3
"""CLI entry for the isolated long-term memory maintenance v1.

Contract:
    implementation/run_update.py --state-dir <dir> --batch <JSON> --output <JSON path>

Real model calls require DEEPSEEK_API_KEY to be supplied explicitly through the
process environment. This launcher only wires sys.path: it never sets
HOME / CODEX_HOME and never reads or prints credentials.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent                      # .../memory-maintenance-v1/implementation
PROTO_SRC = HERE.parent / "vendor"     # packaged legacy client dependency

sys.path.insert(0, str(HERE / "src"))
if PROTO_SRC.exists():
    sys.path.insert(0, str(PROTO_SRC))

from memmaint.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
