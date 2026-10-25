"""Prove the deployed 8788 assembly equals the tested A/B candidate input, and that
the layer is optional (no stable.md -> behaviour identical to the previous build)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEMO = REPO / "work" / "generated-memory-demo"
EXP = REPO / "work" / "8788-stable-memory-experiment"
sys.path.insert(0, str(DEMO / "src"))

from lettaproto.config import Config, load_config  # noqa: E402
from lettaproto.context import assemble_memory, load_stable_layer, PREAMBLE  # noqa: E402

FULL = (EXP / "backup" / "idea-prod-pre-change.md").read_text("utf-8")
STABLE = (EXP / "stable" / "stable.md").read_text("utf-8")
TESTED = ("# 稳定记忆层（stable.md，实验候选）\n\n" + STABLE
          + "\n\n# 完整事件摘要（原 idea.md，逐字保留）\n\n" + FULL)

cfg = load_config()
asm = assemble_memory(cfg, "prod")
assert asm["stable_present"] is True, "stable layer not detected"
assert asm["text"] == TESTED, "deployed assembly != tested candidate input"
print("OK assembly == tested candidate input; injected chars =", len(asm["text"]))
print("stable chars", len(asm["stable"]), "event chars", len(asm["event"]))
print("event sha256_prefix", hashlib.sha256(asm["event"].encode()).hexdigest()[:12], "(want eb55ce8eff1e)")

# Optional-layer compatibility: a memory dir with idea.md but NO stable.md behaves as before.
import tempfile

tmp = Path(tempfile.mkdtemp())
(tmp / "prod").mkdir(parents=True)
(tmp / "prod" / "idea.md").write_text(FULL, "utf-8")
cfg2 = Config()
cfg2.project_root = tmp
cfg2.memory_root = tmp
cfg2.ensure_dirs()
asm2 = assemble_memory(cfg2, "prod")
assert asm2["stable_present"] is False
assert asm2["text"] == FULL, "without stable.md the injected memory must be idea.md only"
assert load_stable_layer(cfg2, "prod") is None
print("OK no-stable-layer compat: injected == idea.md only")
print("ALL ASSEMBLY CHECKS PASSED")
