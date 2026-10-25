"""Non-secret configuration and path resolution.

Secrets are NEVER stored here. The DeepSeek key is read only from the process
environment (injected by the trusted credential wrapper).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value else default


@dataclass
class Config:
    project_root: Path = PROJECT_ROOT
    # ---- real export source (read-only) ----
    export_dir: Path = field(
        default_factory=lambda: Path(
            _env(
                "LETTAPROTO_EXPORT_DIR",
                str(PROJECT_ROOT / "inputs" / "exports"),
            )
        )
    )
    # ---- derived data (independent of Letta) ----
    db_path: Path = field(default_factory=lambda: PROJECT_ROOT / "db" / "canonical.sqlite")
    exports_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "exports")
    runs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "runs")
    # ---- working memory (idea.md): the user's chosen memory snapshot ----
    memory_root: Path = field(default_factory=lambda: PROJECT_ROOT / "memory")
    source_idea: Path = field(
        default_factory=lambda: Path(
            _env("LETTAPROTO_SOURCE_IDEA", str(PROJECT_ROOT / "inputs" / "idea.md"))
        )
    )
    # ---- Letta runtime (reuses validated local install, read-only) ----
    letta_bin: Path = field(
        default_factory=lambda: Path(
            _env("LETTAPROTO_LETTA_BIN", str(PROJECT_ROOT.parent / "letta-validation" / "node_modules" / ".bin" / "letta"))
        )
    )
    # vendor patch target: a project-local copy of the bundle, patched so hardcoded
    # home paths follow LETTA_HOME instead of $HOME/.letta (the read-only install and
    # the user's ~/.letta are never touched).
    letta_src_pkg: Path = field(
        default_factory=lambda: PROJECT_ROOT.parent / "letta-validation" / "node_modules" / "@letta-ai" / "letta-code"
    )
    vendor_root: Path = field(default_factory=lambda: PROJECT_ROOT / "runtime" / "vendor")
    letta_home: Path = field(default_factory=lambda: PROJECT_ROOT / "runtime" / "letta-home")
    # ---- model provider: reuse the already-verified configured endpoint ----
    api_base_url: str = field(default_factory=lambda: _env("LETTAPROTO_API_BASE_URL", "https://api.deepseek.com"))
    chat_model: str = field(default_factory=lambda: _env("LETTAPROTO_CHAT_MODEL", "deepseek-flash"))
    letta_model: str = field(default_factory=lambda: _env("LETTAPROTO_LETTA_MODEL", "deepseek/deepseek-flash"))
    api_key_env: str = "DEEPSEEK_API_KEY"
    # ---- budgets (engineering caps, not per-request context limits) ----
    max_requests: int = 30
    max_input_tokens: int = 500_000
    # user-trial quota beyond the dev cap; cumulative usage is preserved
    trial_max_requests: int = 60
    # updater input ceiling: over this, FAIL loudly instead of silently truncating
    updater_max_input_tokens: int = 200_000
    # chat continuity defaults (prototype chat_log; NOT the imported export history)
    default_recent_messages: int = 40
    default_recent_token_budget: int = 60_000
    # agentic chat + background worker
    chat_tool_round_limit: int = 4
    chat_max_output_tokens: int = 16_000
    chat_final_max_output_tokens: int = 32_000
    search_round_limit: int = 2
    worker_poll_seconds: float = 2.0
    worker_max_attempts: int = 3
    worker_idle_enabled: bool = True
    # ---- budget counters persisted by deepseek client ----
    usage_file: Path = field(
        default_factory=lambda: Path(_env("LETTAPROTO_USAGE_FILE", str(PROJECT_ROOT / "runs" / "api_usage.json")))
    )

    def load_overrides(self) -> "Config":
        cfg_file = self.project_root / "config.json"
        if cfg_file.exists():
            data = json.loads(cfg_file.read_text("utf-8"))
            for key, value in data.items():
                if hasattr(self, key) and value is not None:
                    cur = getattr(self, key)
                    setattr(self, key, Path(value) if isinstance(cur, Path) else value)
        return self

    @property
    def vendor_pkg(self) -> Path:
        return self.vendor_root / "letta-code"

    @property
    def letta_entry(self) -> Path:
        """Prefer the vendored, patched bundle; fall back to the read-only install."""
        vendored = self.vendor_pkg / "letta.js"
        return vendored if vendored.exists() else self.letta_bin

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env)

    def ensure_dirs(self) -> None:
        for path in (
            self.db_path.parent,
            self.exports_dir,
            self.runs_dir,
            self.letta_home,
            self.vendor_root,
            self.memory_root,
            self.memory_root / "prod" / "versions",
            self.memory_root / "test" / "versions",
        ):
            path.mkdir(parents=True, exist_ok=True)


def load_config() -> Config:
    config = Config().load_overrides()
    config.ensure_dirs()
    return config
