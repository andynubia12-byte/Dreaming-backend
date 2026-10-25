#!/usr/bin/env python3
"""Single CLI entry to import a read-only retrieval history into the trial.

    python3 import_history.py --source-db <retrieval_import.sqlite> --manifest <manifest.json>

0 API. The source sqlite is opened read-only and never modified. Only the trial's
own database is written (``messages``/``conversations`` + import metadata); the
imported history is not written to ``chat_log`` and not enqueued for maintenance.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from continuity.history import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
