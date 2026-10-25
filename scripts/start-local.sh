#!/bin/sh
# No credential discovery, dependency installs, or API calls at startup.
set -eu
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
TRIAL="$ROOT/work/continuity-trial-v2"
MODE=workbench
if [ "${1:-}" = "--trial" ]; then
    MODE=trial
    shift
fi
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$TRIAL/src"
cd "$TRIAL"
if [ "$MODE" = trial ]; then
    exec "${PYTHON:-python3}" "$TRIAL/start.py" --port 8794 "$@"
fi
exec "${PYTHON:-python3}" "$TRIAL/start_workbench.py" --port 8794 "$@"
