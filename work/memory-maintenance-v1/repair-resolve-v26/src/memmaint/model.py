"""Model access for the maintenance loop.

* DeepSeekModel — metered Flash calls via the READ-ONLY prototype client. The
  API key is read only from the environment injected by the trusted wrapper; it
  is never written to disk, argv, or logs. Responses are validated STRICTLY:
  `finish_reason == "length"`, tool calls, leaked tool-call serialization, and
  invalid/truncated JSON are all rejected (no truncation repair). Every attempt
  (success or failure) is logged with its usage to calls.jsonl; the independent
  ledger (usage.json) counts every request the client issued.
* ScriptedModel — deterministic responses for the mock test suite (no network).
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable


class ModelError(RuntimeError):
    """Raised when a model stage cannot produce usable output."""


_TOOL_MARKERS = (
    "\uff5c\uff5cDSML\uff5c\uff5c",  # full-width vertical bars around DSML
    "<|DSML|>", "<|dsml|>", "<|tool_calls|>", "</tool_calls>", "<tool_calls>", "<|tool_call|>",
)


def _looks_like_tool_serialization(text: str | None) -> bool:
    try:
        from lettaproto.util import looks_like_tool_serialization as f
        return bool(f(text))
    except Exception:  # noqa: BLE001
        return bool(text) and any(marker in text for marker in _TOOL_MARKERS)


def _strict_json(text: str) -> Any:
    """Parse model JSON strictly: raw JSON or a fenced block, no truncation repair."""
    body = (text or "").strip()
    if not body:
        raise ModelError("empty model content")
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass
    for fence in ("```json", "```"):
        if fence in body:
            segment = body.split(fence, 1)[1].split("```", 1)[0].strip()
            try:
                return json.loads(segment)
            except json.JSONDecodeError:
                continue
    raise ModelError("invalid JSON (strict parse; truncation repair disabled)")


def _validate_shape(value: Any, keys: list[str]) -> dict:
    """A syntactically valid JSON value still must be the expected object shape."""
    if not isinstance(value, dict):
        raise ModelError(f"expected a JSON object, got {type(value).__name__}")
    for key in keys:
        if key not in value:
            raise ModelError(f"missing required key '{key}'")
        if not isinstance(value[key], list):
            raise ModelError(f"key '{key}' must be a JSON array")
    return value


def _is_index(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_operations(value: Any) -> dict:
    """Each `operations` element must itself be a well-formed object."""
    _validate_shape(value, ["operations"])
    for element in value["operations"]:
        if not isinstance(element, dict):
            raise ModelError("operation element is not a JSON object")
        if not isinstance(element.get("op"), str):
            raise ModelError("operation element missing string 'op'")
        if not isinstance(element.get("text"), str):
            raise ModelError("operation element missing string 'text'")
    return value


def _validate_verdicts(value: Any, count: int) -> dict:
    """Each `verdicts` element must be well-formed and cover exactly the candidates."""
    _validate_shape(value, ["verdicts"])
    seen: set[int] = set()
    for element in value["verdicts"]:
        if not isinstance(element, dict):
            raise ModelError("verdict element is not a JSON object")
        index = element.get("index")
        if not _is_index(index):
            raise ModelError("verdict element missing/invalid integer index")
        if element.get("verdict") not in ("supported", "partial", "unsupported"):
            raise ModelError(f"verdict element has invalid verdict: {element.get('verdict')!r}")
        for flag in ("role_ok", "time_ok", "overgeneralized"):
            if not isinstance(element.get(flag), bool):
                raise ModelError(f"verdict element missing boolean '{flag}'")
        if index in seen:
            raise ModelError(f"duplicate verdict index {index}")
        seen.add(index)
    if seen != set(range(count)):
        raise ModelError(f"verdict indices {sorted(seen)} do not cover all candidates")
    return value


def _parse_http_code(exc: BaseException) -> int | None:
    match = re.search(r"HTTP (\d{3})", str(exc))
    return int(match.group(1)) if match else None


def _is_permanent(exc: BaseException) -> bool:
    """Permanent request errors (bad key, no balance, bad request) must not retry."""
    code = _parse_http_code(exc)
    if code is None:
        return False           # network/timeout -> retryable
    if code >= 500:
        return False
    if code in (408, 425, 429):
        return False
    return True


def _append_log(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")


class Model:
    def propose(self, system: str, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def verify(self, system: str, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def summarize(self, system: str, payload: dict[str, Any]) -> str:
        raise NotImplementedError

    def call_count(self) -> int:
        return 0


class DeepSeekModel(Model):
    def __init__(
        self,
        ledger_dir: Path,
        *,
        model: str = "deepseek-flash",
        base_url: str = "https://api.deepseek.com",
        max_requests: int = 100_000,
        max_retries: int = 1,
        max_output_tokens: int = 16_000,
    ) -> None:
        from lettaproto.config import Config as PConfig  # lazy: mock runs need no prototype
        from lettaproto.deepseek import DeepSeekClient

        self.ledger_dir = Path(ledger_dir)
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        self.usage_file = self.ledger_dir / "usage.json"
        self.call_log = self.ledger_dir / "calls.jsonl"

        cfg = PConfig()
        cfg.usage_file = self.usage_file
        cfg.chat_model = model
        cfg.api_base_url = base_url
        cfg.max_requests = max_requests
        cfg.trial_max_requests = max_requests
        cfg.max_input_tokens = 10 ** 12
        self._client = DeepSeekClient(cfg)
        self.model = model
        self.max_retries = max_retries
        self.max_output_tokens = max_output_tokens
        self._lock = threading.Lock()
        self._count = 0

    def call_count(self) -> int:
        return self._count

    def _log(self, kind: str, attempt: int, started: float, ok: bool,
             usage: dict | None, error: str | None, finish: str | None,
             content: str | None = None) -> None:
        text = content if isinstance(content, str) else None
        original_len = len(text) if text is not None else 0
        truncated = text is not None and original_len > 20000
        _append_log(self.call_log, {
            "type": "api", "kind": kind, "model": self.model, "attempt": attempt,
            "elapsed_s": round(time.time() - started, 2), "ok": ok, "finish_reason": finish,
            "usage": usage, "error": error,
            "content": (text[:20000] if truncated else text),
            "content_len": original_len, "content_truncated": truncated,
        })

    def _chat(self, kind: str, system: str, payload: dict[str, Any], max_tokens: int,
              *, json_mode: bool, expect: Callable[[Any], Any] | None = None):
        user = json.dumps(payload, ensure_ascii=False)
        last: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            started = time.time()
            usage: dict | None = None
            finish: str | None = None
            raw: str | None = None
            try:
                resp = self._client.chat(
                    [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    max_tokens=min(max_tokens, self.max_output_tokens), temperature=0.2,
                    response_json=json_mode, model=self.model,
                )
                usage = resp.get("usage") or {}
                choice = (resp.get("choices") or [{}])[0]
                finish = choice.get("finish_reason")
                message = choice.get("message") or {}
                raw = message.get("content") or ""
                if finish != "stop":
                    raise ModelError(f"unexpected finish_reason={finish!r}")
                if message.get("tool_calls"):
                    raise ModelError("unexpected tool_calls in response")
                text = raw.strip()
                if not text:
                    raise ModelError("empty model content")
                if _looks_like_tool_serialization(text):
                    raise ModelError("leaked tool-call serialization in content")
                value = _strict_json(text) if json_mode else text
                if expect is not None:
                    expect(value)
                with self._lock:
                    self._count += 1
                self._log(kind, attempt, started, True, usage, None, finish, content=raw)
                return value
            except BaseException as exc:  # noqa: BLE001
                last = exc
                self._log(kind, attempt, started, False, usage,
                          f"{type(exc).__name__}: {str(exc)[:200]}", finish, content=raw)
                if _is_permanent(exc):
                    raise ModelError(f"{kind} permanent request error (no retry): {exc}") from exc
        raise ModelError(f"{kind} failed after {self.max_retries + 1} attempts: {last}")

    def propose(self, system: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._chat("propose", system, payload, self.max_output_tokens, json_mode=True,
                          expect=_validate_operations)

    def verify(self, system: str, payload: dict[str, Any]) -> dict[str, Any]:
        count = len(payload.get("claims") or [])
        return self._chat("verify", system, payload, self.max_output_tokens, json_mode=True,
                          expect=lambda v: _validate_verdicts(v, count))

    def summarize(self, system: str, payload: dict[str, Any]) -> str:
        return self._chat("summarize", system, payload, self.max_output_tokens, json_mode=False)


def _default_verify(_system: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"verdicts": [
        {"index": c["index"], "verdict": "supported", "role_ok": True, "time_ok": True,
         "overgeneralized": False, "issue": "", "corrected_text": None}
        for c in payload.get("claims", [])
    ]}


def _default_summarize(_system: str, payload: dict[str, Any]) -> str:
    return "# 当前理解（由记忆材料重建）\n\n" + payload.get("memory_material", "")


class ScriptedModel(Model):
    """Deterministic model for tests / offline demos."""

    def __init__(
        self,
        propose: Callable[[str, dict], dict] | None = None,
        verify: Callable[[str, dict], dict] | None = None,
        summarize: Callable[[str, dict], str] | None = None,
    ) -> None:
        self._propose = propose or (lambda _s, _p: {"operations": []})
        self._verify = verify or _default_verify
        self._summarize = summarize or _default_summarize
        self.calls: list[str] = []

    def propose(self, system: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("propose")
        return self._propose(system, payload)

    def verify(self, system: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("verify")
        return self._verify(system, payload)

    def summarize(self, system: str, payload: dict[str, Any]) -> str:
        self.calls.append("summarize")
        return self._summarize(system, payload)

    def call_count(self) -> int:
        return len(self.calls)
