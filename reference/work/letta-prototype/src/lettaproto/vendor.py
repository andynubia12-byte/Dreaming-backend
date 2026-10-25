"""Vendor-patch the Letta bundle so hardcoded home paths follow LETTA_HOME.

Rationale: Letta Code 0.34.10 resolves its user settings/permissions/remote-state
under `os.homedir()/.letta` with no supported path-override parameter (verified by
source inspection). Reassigning HOME is disallowed by project policy, and the
read-only install (`work/letta-validation`) and the user's `~/.letta` must not be
touched. So we copy the bundle into the project runtime and apply a minimal,
explicit patch: every patched site prefers `LETTA_HOME` when it is set.

Only the sites our flows actually write are patched. The patch is recorded in
`patch.json` (original sha256, patched sha256, applied replacements) and is
verified after application.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from .config import Config
from .util import read_json, sha256_file, write_json

MARKER = "// VENDOR PATCH (letta-prototype): home paths follow LETTA_HOME. Do not edit upstream."

# Exact textual replacements. Each is verified to occur exactly once before use.
REPLACEMENTS: list[tuple[str, str]] = [
    (
        'const dirPath = join5(home, ".letta");',
        'const dirPath = process.env.LETTA_HOME || join5(home, ".letta");',
    ),
    (
        'return join5(home, ".letta", "settings.json");',
        'return process.env.LETTA_HOME ? join5(process.env.LETTA_HOME, "settings.json") : join5(home, ".letta", "settings.json");',
    ),
    (
        'return path11.join(getRemoteSettingsHome(), ".letta", "remote-settings.json");',
        'return process.env.LETTA_HOME ? path11.join(process.env.LETTA_HOME, "remote-settings.json") : path11.join(getRemoteSettingsHome(), ".letta", "remote-settings.json");',
    ),
    (
        'const legacyPath = path11.join(getRemoteSettingsHome(), ".letta", "cwd-cache.json");',
        'const legacyPath = process.env.LETTA_HOME ? path11.join(process.env.LETTA_HOME, "cwd-cache.json") : path11.join(getRemoteSettingsHome(), ".letta", "cwd-cache.json");',
    ),
    (
        'canonical: join53(homeDir, ".letta", "settings.json"),',
        'canonical: process.env.LETTA_HOME ? join53(process.env.LETTA_HOME, "settings.json") : join53(homeDir, ".letta", "settings.json"),',
    ),
    (
        'legacy: join53(xdgConfigHome, "letta", "settings.json")',
        'legacy: process.env.LETTA_HOME ? join53(process.env.LETTA_HOME, "settings.json") : join53(xdgConfigHome, "letta", "settings.json")',
    ),
    (
        'return join41(process.env.HOME ?? process.env.USERPROFILE ?? "~", ".letta");',
        'return process.env.LETTA_HOME ?? join41(process.env.HOME ?? process.env.USERPROFILE ?? "~", ".letta");',
    ),
    (
        'return join106(homedir57(), ".letta", "settings.json");',
        'return process.env.LETTA_HOME ? join106(process.env.LETTA_HOME, "settings.json") : join106(homedir57(), ".letta", "settings.json");',
    ),
]


def patch_meta_path(config: Config) -> Path:
    return config.vendor_root / "patch.json"


def _source_digest(config: Config) -> str:
    return hashlib.sha256(str(config.letta_src_pkg).encode()).hexdigest()[:16] + ":" + str(
        (config.letta_src_pkg / "letta.js").stat().st_size
    )


def is_vendored(config: Config) -> bool:
    meta = read_json(patch_meta_path(config), None)
    if not meta:
        return False
    vendored = config.vendor_pkg / "letta.js"
    return vendored.exists() and meta.get("source_digest") == _source_digest(config) and meta.get("ok")


def vendor_letta(config: Config, force: bool = False) -> dict[str, Any]:
    src_pkg = config.letta_src_pkg
    if not (src_pkg / "letta.js").exists():
        raise FileNotFoundError(f"source bundle not found: {src_pkg}")
    if is_vendored(config) and not force:
        return {"ok": True, "action": "kept", **read_json(patch_meta_path(config), {})}

    if config.vendor_pkg.exists():
        shutil.rmtree(config.vendor_pkg)
    config.vendor_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_pkg, config.vendor_pkg, symlinks=True)

    # make hoisted dependencies resolvable without touching the read-only install
    link = config.vendor_root / "node_modules"
    if not link.exists():
        link.symlink_to(config.letta_src_pkg.parent.parent)  # -> letta-validation/node_modules

    target = config.vendor_pkg / "letta.js"
    original_sha = sha256_file(target)
    text = target.read_text("utf-8")
    applied: list[dict[str, Any]] = []
    for old, new in REPLACEMENTS:
        count = text.count(old)
        applied.append({"old": old[:70], "occurrences": count})
        if count != 1:
            raise RuntimeError(f"unexpected occurrence count {count} for patch site: {old[:70]}")
        text = text.replace(old, new)
    # marker after shebang
    if text.startswith("#!"):
        first_nl = text.index("\n") + 1
        text = text[:first_nl] + MARKER + "\n" + text[first_nl:]
    else:
        text = MARKER + "\n" + text
    target.write_text(text, "utf-8")
    patched_sha = sha256_file(target)

    meta = {
        "ok": True,
        "action": "patched",
        "source_pkg": str(src_pkg),
        "source_digest": _source_digest(config),
        "original_sha256": original_sha,
        "patched_sha256": patched_sha,
        "replacements": applied,
        "marker": MARKER,
        "note": "Letta 0.34.10 exposes no supported settings-path override; patched sites prefer LETTA_HOME.",
    }
    write_json(patch_meta_path(config), meta)
    return meta


def verify(config: Config) -> dict[str, Any]:
    target = config.vendor_pkg / "letta.js"
    if not target.exists():
        return {"ok": False, "error": "vendored bundle missing"}
    text = target.read_text("utf-8")
    checks = {
        "marker_present": MARKER in text,
        "settings_path_patched": 'process.env.LETTA_HOME ? join5(process.env.LETTA_HOME, "settings.json")' in text,
        "persist_dir_patched": 'const dirPath = process.env.LETTA_HOME || join5(home, ".letta");' in text,
        "remote_patched": 'process.env.LETTA_HOME ? path11.join(process.env.LETTA_HOME, "remote-settings.json")' in text,
        "permissions_patched": 'canonical: process.env.LETTA_HOME ? join53(process.env.LETTA_HOME, "settings.json")' in text,
        "unpatched_home_settings_remain": 'return join5(home, ".letta", "settings.json");' in text,
    }
    meta = read_json(patch_meta_path(config), {}) or {}
    checks["sha_matches"] = meta.get("patched_sha256") == sha256_file(target)
    checks["ok"] = all(
        checks[k]
        for k in ("marker_present", "settings_path_patched", "persist_dir_patched", "remote_patched", "permissions_patched", "sha_matches")
    )
    return checks
