"""Non-secret configuration for the isolated continuity trial.

Adapted from ``work/generated-memory-demo/src/lettaproto/config.py``
(sha256 f024ed1962cf5bfe9302af72b47f74e069479015d8b8d81e679c8fc7ef38f2b7).

Every path points inside this directory tree, so the trial never reads or writes
the live 8788 service, the other demos, or the original exports. The DeepSeek key
is read ONLY from an explicitly supplied process environment variable.
Startup does not discover credentials.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

# config.py = <root>/src/continuity/config.py -> parents[2] is the trial root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Packaged legacy v2.6 dependency; this is not the v1 selected summary method.
DEFAULT_MAINT_CLI = (PROJECT_ROOT.parent / "memory-maintenance-v1" /
                     "review-v2" / "run_update_with_citation_alias.py")


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value else default


@dataclass
class Config:
    project_root: Path = PROJECT_ROOT
    # ---- isolated data paths (all inside this tree) ----
    db_path: Path = field(default_factory=lambda: PROJECT_ROOT / "db" / "chat.sqlite")
    runs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "runs")
    maint_state_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "state" / "maintenance")
    # ---- single user, shared long-term memory; sessions isolate recent chat ----
    user_id: str = field(default_factory=lambda: _env("CONTINUITY_USER_ID", "trial-user"))
    instance: str = field(default_factory=lambda: _env("CONTINUITY_INSTANCE", "trial"))
    # optional seam for a future history importer; left unset on purpose
    export_dir: Path | None = None
    # read-only retrieval import source (history-bootstrap-v1 output). The importer
    # never reads the raw export dir; it only reads this independent sqlite+manifest.
    retrieval_source_db: Path | None = None
    retrieval_manifest: Path | None = None
    # ---- model provider (reuses the verified DeepSeek-compatible endpoint) ----
    api_base_url: str = field(default_factory=lambda: _env("CONTINUITY_API_BASE_URL", "https://api.deepseek.com"))
    # retrieval / consolidation agent model stays Flash (never silently upgraded)
    chat_model: str = field(default_factory=lambda: _env("CONTINUITY_RETRIEVAL_MODEL", "deepseek-flash"))
    # user-facing main chat runs on Pro
    main_chat_model: str = field(default_factory=lambda: _env("CONTINUITY_MAIN_MODEL", "deepseek-v4-pro"))
    api_key_env: str = "DEEPSEEK_API_KEY"
    # ---- maintenance CLI wiring (engine NOT modified) ----
    maint_cli: Path = field(default_factory=lambda: DEFAULT_MAINT_CLI)
    maint_model: str = field(default_factory=lambda: _env("CONTINUITY_MAINT_MODEL", "deepseek-flash"))
    # TOTAL trial budget for the maintenance engine. The engine's --budget is a
    # CUMULATIVE cap stored in its ledger (state/maintenance/ledger/usage.json), not
    # a per-invocation cap, so this number spans the whole trial. ~3 model calls per
    # changed turn -> 180 ~= 60 turns. It is never auto-reset or auto-increased.
    maint_total_budget: int = field(default_factory=lambda: int(_env("CONTINUITY_MAINT_BUDGET", "180")))
    maint_expected_calls_per_turn: int = 3
    # Output headroom for the maintenance CLI call (``--output-tokens``). The CLI's
    # own default is a small 16k that truncates long consolidation results; the
    # trial's daily queue must be as generous as the 64k used by the personal
    # bootstrap, so it passes this value explicitly on every invocation.
    maint_max_output_tokens: int = field(
        default_factory=lambda: int(_env("CONTINUITY_MAINT_MAX_OUTPUT_TOKENS", "64000")))
    # A full consolidation turn can call the model several times; 900s is a
    # generous whole-invocation ceiling (configurable via env / config.json).
    maint_timeout_seconds: int = field(
        default_factory=lambda: int(_env("CONTINUITY_MAINT_TIMEOUT", "900")))
    # ---- durable serial maintenance queue ----
    queue_poll_seconds: float = 1.0
    queue_max_attempts: int = 3
    # ---- model budgets (engineering caps, not context limits) ----
    max_requests: int = 30
    max_input_tokens: int = 2_000_000
    trial_max_requests: int = 120
    # ---- chat context defaults ----
    default_recent_messages: int = 40
    default_recent_token_budget: int = 60_000
    chat_tool_round_limit: int = 4
    chat_max_output_tokens: int = 16_000
    chat_final_max_output_tokens: int = 32_000
    search_round_limit: int = 2
    # ---- usage counters (separate from the maintenance engine ledger) ----
    usage_file: Path = field(
        default_factory=lambda: PROJECT_ROOT / "runs" / "api_usage.json"
    )

    def load_overrides(self) -> "Config":
        cfg_file = self.project_root / "config.json"
        if cfg_file.exists():
            data = json.loads(cfg_file.read_text("utf-8"))
            for key, value in data.items():
                if hasattr(self, key) and (value is not None or key in {"export_dir", "retrieval_source_db", "retrieval_manifest"}):
                    path_fields = {"db_path", "runs_dir", "maint_state_dir", "usage_file",
                                   "maint_cli", "export_dir", "retrieval_source_db", "retrieval_manifest"}
                    if key == "project_root":
                        continue  # package location follows this source file
                    if key in path_fields and value is not None:
                        value = (self.project_root / value).resolve()
                        if not value.is_relative_to(PROJECT_ROOT.parent.parent):
                            raise ValueError(f"{key} must stay inside the packaged project")
                    setattr(self, key, value)
        return self

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env)

    def ensure_dirs(self) -> None:
        for path in (self.db_path.parent, self.runs_dir, self.maint_state_dir):
            path.mkdir(parents=True, exist_ok=True)


def load_config() -> Config:
    # NOTE: callers apply their own path overrides first, then call ensure_dirs()
    # (so a --db/--state-dir override does not leave stray default dirs behind).
    return Config().load_overrides()
