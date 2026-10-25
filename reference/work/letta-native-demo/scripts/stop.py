#!/usr/bin/env python3
"""Stop the demo server (and its Letta session) by PID."""

from __future__ import annotations

import os
import signal
from pathlib import Path

ROUTE = Path(__file__).resolve().parent.parent
PID_FILE = ROUTE / "runs" / "server.pid"


def main() -> None:
    if not PID_FILE.exists():
        print("no pid file")
        return
    pid = int(PID_FILE.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"stopped pid={pid}")
    except ProcessLookupError:
        print(f"pid {pid} not running")


if __name__ == "__main__":
    main()
