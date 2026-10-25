#!/bin/sh
set -eu
HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON:-python3}" "$HERE/run_update.py" "$@"
