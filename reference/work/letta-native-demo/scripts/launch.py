#!/usr/bin/env python3
"""Start the demo server in its own session so it survives after this task ends."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROUTE = Path(__file__).resolve().parent.parent
PID_FILE = ROUTE / "runs" / "server.pid"
LOG = ROUTE / "logs" / "server.log"


def main() -> None:
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log = LOG.open("ab")
    proc = subprocess.Popen(
        [sys.executable, str(ROUTE / "scripts" / "server.py")],
        stdin=subprocess.DEVNULL, stdout=log, stderr=log,
        start_new_session=True, cwd=str(ROUTE),
        env={**__import__("os").environ},
    )
    PID_FILE.write_text(str(proc.pid))
    print(f"started pid={proc.pid}")


if __name__ == "__main__":
    main()
