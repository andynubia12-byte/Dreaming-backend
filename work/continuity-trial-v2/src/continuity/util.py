"""Small shared helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
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


# Local trial adaptation: one classifier shared by final validation and all
# assistant read filters. Keep this set in sync with chat/Memory Agent schemas
# (offline contract test); ordinary HTML/math tags are deliberately not banned.
REGISTERED_TOOL_NAMES = (
    'delegate_memory', 'search_memory', 'search_history', 'read_message', 'expand',
)
_XML_TOOL_NAMES = REGISTERED_TOOL_NAMES + ('tool_calls', 'tool_call')
_TOOL_MARKER_TOKENS = (
    '｜｜DSML｜｜', '<|DSML|>', '<|dsml|>', '<|tool_calls|>',
    '</tool_calls>', '<tool_calls>', '<|tool_call|>',
)
_XML_NAMES_PATTERN = '(?:' + '|'.join(map(re.escape, _XML_TOOL_NAMES)) + ')'
_XML_TOOL = re.compile(r'<\s*/?\s*' + _XML_NAMES_PATTERN + r'(?=[\s/>])', re.I)
_XML_TOOL_EOF = re.compile(r'<\s*/?\s*' + _XML_NAMES_PATTERN + r'$', re.I)
_DSML = re.compile(r'[|｜]{1,2}\s*dsml\s*[|｜]{1,2}', re.I)
_PIPE_TOOL = re.compile(r'<\s*[|｜]\s*(?:dsml|tool_calls|tool_call)\s*[|｜]\s*>', re.I)


def looks_like_tool_serialization(text: str | None, *, final: bool = True) -> bool:
    """Recognize reserved tool envelopes, including fenced/mixed-case XML.

    No XML is interpreted or dispatched. `final=False` waits for a name boundary
    so a chunk ending in `<expand` does not prematurely reject `<expanded>`.
    """
    if not text:
        return False
    folded = text.casefold()
    return bool(any(token.casefold() in folded for token in _TOOL_MARKER_TOKENS)
                or _DSML.search(text) or _PIPE_TOOL.search(text) or _XML_TOOL.search(text)
                or (final and _XML_TOOL_EOF.search(text)))


def tool_serialization_pending_suffix(text: str) -> int:
    """Suffix that could still become a reserved marker after the next delta."""
    keep = 0
    folded = text.casefold()
    for marker in _TOOL_MARKER_TOKENS:
        for n in range(1, len(marker)):
            if folded.endswith(marker[:n].casefold()):
                keep = max(keep, n)
    xml = re.search(r'<\s*/?\s*([a-z_]*)$', text, re.I)
    if xml and any(name.startswith(xml[1].casefold()) for name in _XML_TOOL_NAMES):
        keep = max(keep, len(text) - xml.start())
    pipe = re.search(r'<\s*[|｜][^>]*$', text)
    if pipe:
        keep = max(keep, len(text) - pipe.start())
    dsml = re.search(r'[|｜]{1,2}\s*([a-z]*)\s*$', text, re.I)
    if dsml and 'dsml'.startswith(dsml[1].casefold()):
        keep = max(keep, len(text) - dsml.start())
    return keep


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
