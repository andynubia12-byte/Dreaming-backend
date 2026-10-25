"""Frozen maintenance CLI plus the declared summary-only reference alias wire."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BASE / 'repair-resolve-v26/src'),
               str(BASE / 'vendor')]
from citation_alias_wire import install
from memmaint.cli import main


if __name__ == '__main__':
    install()
    # Optional side channel; default invocation keeps exactly the original path.
    import os
    if os.environ.get('CONTINUITY_OBSERVER_FD'):
        from realtime_observer import install as install_observer
        install_observer()
    raise SystemExit(main())
