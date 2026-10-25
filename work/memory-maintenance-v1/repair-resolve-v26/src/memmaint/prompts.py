"""Prompt loading and hashing (the launched engine freezes these hashes)."""

from __future__ import annotations

import hashlib
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"  # implementation/prompts
NAMES = ("propose", "verify", "summarize")


def load_prompts() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in NAMES:
        out[name] = (PROMPT_DIR / f"{name}.md").read_text("utf-8").strip()
    return out


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_hashes(prompts: dict[str, str]) -> dict[str, str]:
    return {name: sha256_text(text) for name, text in prompts.items()}
