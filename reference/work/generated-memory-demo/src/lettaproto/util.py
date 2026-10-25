"""Small shared helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# Internal tool-call serialization that must NEVER be shown as a user-facing answer
# or replayed as assistant context. DeepSeek's leaked tool markers use full-width
# vertical bars (U+FF5C) around the token "DSML"; we also match the ASCII variant
# and any OpenAI-style raw tool_call envelope. Matching is on the marker tokens only,
# so ordinary prose that merely mentions "DSML"/"tools" is not filtered.
_TOOL_MARKER_TOKENS = (
    "\uff5c\uff5cDSML\uff5c\uff5c",   # \uFF5C = full-width vertical bar
    "<|DSML|>",
    "<|dsml|>",
    "<|tool_calls|>",
    "</tool_calls>",
    "<tool_calls>",
    "<|tool_call|>",
)


def looks_like_tool_serialization(text: str | None) -> bool:
    """True if *text* carries leaked internal tool-call markup (DSML or raw envelope).

    Detection is deliberately marker-based, not keyword-based: only the actual
    serialization tokens trigger it, so a user pasting technical prose is untouched.
    """
    if not text:
        return False
    return any(token in text for token in _TOOL_MARKER_TOKENS)


def approx_tokens(text: str) -> int:
    """Rough token estimate (clearly labelled as an estimate, not a measurement).

    Mixed CJK/latin heuristic: CJK ~1 token/char, other ~1 token/4 chars.
    """
    cjk = 0
    other = 0
    for ch in text:
        if "\u3040" <= ch <= "\u9fff" or "\uff00" <= ch <= "\uffef":
            cjk += 1
        else:
            other += 1
    return cjk + (other + 3) // 4


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), "utf-8")
    tmp.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text("utf-8"))


def text_from_parts(parts: Iterable[Any]) -> str:
    """Join textual parts; skip asset/pointer objects (indexed separately)."""
    out: list[str] = []
    for part in parts:
        if isinstance(part, str):
            out.append(part)
        elif isinstance(part, dict):
            if part.get("content_type") == "text" and isinstance(part.get("text"), str):
                out.append(part["text"])
            elif isinstance(part.get("text"), str):
                out.append(part["text"])
    return "\n".join(p for p in out if p)


def iso_from_epoch(value: Any) -> str | None:
    if value is None:
        return None
    try:
        import datetime as _dt

        return _dt.datetime.fromtimestamp(float(value), _dt.timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def guard_user_letta() -> dict[str, Any]:
    """Snapshot the ambient ~/.letta so we can detect/undo accidental writes."""
    home = Path(os.environ.get("HOME") or Path.home())
    target = home / ".letta"
    snap: dict[str, Any] = {"path": str(target), "existed": target.exists(), "entries": []}
    if target.exists():
        for entry in sorted(target.rglob("*")):
            if entry.is_file():
                snap["entries"].append({"rel": str(entry.relative_to(target)), "size": entry.stat().st_size})
    return snap
