"""Read-only Codex subscription credentials, resolved only at request time.

Adapted from the original llmtest/server.mjs:
loadAuthFile (234–254), getCodexAuth expiry check (334–338), and
codexHeaders (471–482); source SHA-256:
d67a84965dc1b0af45b4ffbbed7c163692bd0765ae0783a46eac12555537df8f.
No refresh, login, configuration lookup, or credential writes are performed.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import time
import uuid


class CodexCredentialError(RuntimeError):
    """Credentials are unavailable, invalid, or expired; safe to display."""


def validate_codex_base_url(base_url: str) -> None:
    """Validate an exact subscription URL without resolving any credentials."""
    if base_url not in (
        "https://chatgpt.com/backend-api/codex",
        "https://chatgpt.com/backend-api/codex/responses",
    ):
        raise ValueError("Codex subscription requires the fixed ChatGPT backend URL")


def _header_value(value: object) -> bool:
    return isinstance(value, str) and bool(value) and all(
        33 <= ord(char) <= 126 for char in value
    )


def _expired(token: str) -> bool:
    parts = token.split(".")
    if len(parts) != 3:
        return False
    try:
        payload = base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4))
        claims = json.loads(payload)
        exp = claims["exp"]
        if isinstance(exp, bool):
            return False
        return float(exp) <= time.time()
    except (ValueError, TypeError, KeyError):
        return False  # Opaque/unparseable credentials must fail at the upstream.


def resolve_codex_headers(base_url: str, *, auth_path: Path | None = None) -> dict[str, str]:
    """Read credentials for an allowed endpoint; never refresh or persist them.

    Providers must call this only for a real request, and must keep these headers
    out of logs/results. An opaque token is allowed; upstream 401 is a failure.
    """
    validate_codex_base_url(base_url)
    if auth_path is None:
        from .config import PROJECT_ROOT
        # Explicit opt-in uses only this new package's credential directory.
        auth_path = PROJECT_ROOT / "secrets" / "codex-auth.json"
    try:
        auth = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise CodexCredentialError("Codex credentials unavailable; configure the package-local secrets/codex-auth.json explicitly.") from None
    tokens = auth.get("tokens") if isinstance(auth, dict) else None
    if not isinstance(tokens, dict) or not _header_value(tokens.get("access_token")):
        raise CodexCredentialError("Codex credentials invalid; configure the package-local secrets/codex-auth.json explicitly.")
    token = tokens["access_token"]
    account_id = tokens.get("account_id")
    if account_id not in (None, "") and not _header_value(account_id):
        raise CodexCredentialError("Codex credentials invalid; configure the package-local secrets/codex-auth.json explicitly.")
    if _expired(token):
        raise CodexCredentialError("Codex credentials expired; configure the package-local secrets/codex-auth.json explicitly.")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "OpenAI-Beta": "responses=experimental",
        "originator": "codex_cli_rs",
        "session_id": str(uuid.uuid4()),
        "User-Agent": "codex_cli_rs/0.160.1",
    }
    if account_id:
        headers["ChatGPT-Account-Id"] = account_id
    return headers
