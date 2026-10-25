#!/usr/bin/env python3
"""Formal isolated evaluator (stdlib only)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from continuity.evaluation import main

if __name__ == '__main__':
    raise SystemExit(main())
