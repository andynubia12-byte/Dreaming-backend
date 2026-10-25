#!/bin/sh
set -eu
HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$HERE/src"
exec "${PYTHON:-python3}" "$HERE/start_workbench.py" "$@"
