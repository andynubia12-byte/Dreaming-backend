#!/bin/zsh
# Isolation wrapper.
#
# Layer 1: run the trusted credential wrapper with the REAL HOME so it can read
#          the macOS Keychain, and let it inject DEEPSEEK_API_KEY in-memory.
# Layer 2: re-exec python with PYTHONPATH only. HOME is NOT reassigned; Letta's
#          vendored, patched bundle makes hardcoded home paths follow LETTA_HOME
#          (see src/lettaproto/vendor.py) so ~/.letta is never written.
#
# The API key never touches a file, argv, or a log. HOME / CODEX_HOME are never
# reassigned.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SKILL_DIR="${CODEX_DEEPSEEK_SKILL_DIR:-/Users/andy/.codex/skills/codex-deepseek-subagent}"

exec node "$SKILL_DIR/scripts/credential-ui/src/profile.ts" run default -- \
  /usr/bin/env PYTHONPATH="$HERE/src" \
  python3 -m lettaproto.cli "$@"
