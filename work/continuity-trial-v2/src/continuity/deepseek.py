"""Responses-only model client with a hard request/token budget.

Adapted from ``work/generated-memory-demo/src/lettaproto/deepseek.py``
(original sha256 dcca7e33ceca1bc5f5b6eac7a483a6e6eb6d4e2ada796ef5a726f0d0a79becce).

Change (isolated trial only): the cumulative usage ledger is now shared across all
client instances in this process and every "check budget + reserve request" and
"record tokens" step is atomic under one lock, reading the freshest ledger state.
The original per-instance snapshot allowed two concurrent conversations to read the
same old count and overwrite each other, under-counting the promised total budget.

The API key is read ONLY from the process environment (injected by the trusted
credential wrapper). It is never written to disk, argv, or logs.
"""

from __future__ import annotations

import json
import hashlib
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config
from .util import read_json, write_json


class BudgetExceeded(RuntimeError):
    pass


class MissingCredential(RuntimeError):
    pass


# One lock + one shared ledger object per usage-file path: budget check, request
# reservation, and token accounting are all serialized and never lost to a stale
# per-instance snapshot. (Single-process guarantee; the trial runs as one server.)
_USAGE_LOCK = threading.Lock()
_USAGE_STATE: dict[str, "Usage"] = {}


def _usage_key(config: Config) -> str:
    return str(Path(config.usage_file).resolve())


def _coerce_usage(data: Any) -> "Usage":
    data = data if isinstance(data, dict) else {}
    return Usage(
        requests=int(data.get("requests") or 0),
        input_tokens=int(data.get("input_tokens") or 0),
        output_tokens=int(data.get("output_tokens") or 0),
        errors=int(data.get("errors") or 0),
    )


def _shared_usage(config: Config) -> "Usage":
    """The single in-memory ledger for a usage file (call under _USAGE_LOCK)."""
    key = _usage_key(config)
    if key not in _USAGE_STATE:
        _USAGE_STATE[key] = _coerce_usage(read_json(config.usage_file, {}) or {})
    return _USAGE_STATE[key]


def _persist_usage(config: Config) -> None:
    write_json(config.usage_file, _shared_usage(config).to_dict())


def extract_json(text: str) -> Any:
    """Best-effort JSON extraction from model output (fenced / truncated / noisy)."""
    text = (text or "").strip()
    if not text:
        raise ValueError("empty model content")
    candidates: list[str] = [text]
    for fence in ("```json", "```"):
        if fence in text:
            seg = text.split(fence, 1)[1].split("```", 1)[0]
            candidates.append(seg.strip())
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    # truncation repair: try closing at the last few "}" boundaries
    if start >= 0:
        body = text[start:]
        cuts = [i + 1 for i, ch in enumerate(body) if ch == "}"]
        for cut in reversed(cuts[-40:]):
            prefix = body[:cut]
            for suffix in ("", "]", "}]", "}", "]}"):
                try:
                    return json.loads(prefix + suffix)
                except json.JSONDecodeError:
                    continue
    raise ValueError(f"no valid JSON in model output (len={len(text)})")


@dataclass
class Usage:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    errors: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "errors": self.errors,
        }


class DeepSeekClient:
    def __init__(self, config: Config, *, profile=None, recorder=None, role="main"):
        self.config = config
        self.role = role
        self.recorder = recorder
        self._profile_supplied = profile is not None
        from .providers import Profile
        # Even the old unprofiled entry point uses the same stateless Responses
        # transport. There is no brand-specific protocol or fallback route.
        self._provider = Profile(profile if profile is not None else {
            "id": "deepseek-default", "model": config.chat_model,
            "base_url": config.api_base_url, "api_format": "responses",
            "api_key_env": config.api_key_env,
            "parameters": {"reasoning": {"effort": "high"}},
        })
        with _USAGE_LOCK:
            self.usage = _shared_usage(config)

    def _profile_chat(self, messages, *, on_event=None, **options):
        """One Responses transport and ledger for every model and role."""
        from .providers import (ProviderError, captured_lines, read_responses,
                                normalize_response, normalized_usage,
                                redact_record)
        from .provider_stream import StreamProtocolError
        profile = self._provider
        started = time.monotonic()
        event = {"type": "provider_call", "profile_id": profile.id, "role": self.role,
                 "model": options.get("model") or profile.model,
                 "api_format": profile.api_format, "url": profile.url,
                 "auth": profile.auth, "api_key_env": profile.api_key_env,
                 "request": None, "request_body_sha256": None, "request_sent": False,
                 "dispatch_attempted": False, "response_received": False, "sent_unknown": False,
                 "response": None, "events": [], "usage": {}, "http_status": None,
                 "error": None}
        usage, secrets = {}, []
        reserved, failed = False, True
        try:
            body = profile.body(self, messages, stream=on_event is not None, **options)
            event["request"] = body
            event["model"] = body["model"]
            wire = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
            event["request"] = json.loads(wire)
            event["request_body_sha256"] = hashlib.sha256(wire).hexdigest()
            if profile.auth == "codex_subscription":
                from .codex_credentials import resolve_codex_headers, CodexCredentialError
                try:
                    headers = resolve_codex_headers(profile.url)
                except CodexCredentialError as exc:
                    raise MissingCredential(str(exc)) from None
                # Redact bearer and account values if an upstream echoes them.
                for name, value in headers.items():
                    if name.lower() == "authorization":
                        secrets.extend([value, value.removeprefix("Bearer ")])
                    elif name.lower() == "chatgpt-account-id":
                        secrets.append(value)
            else:
                key = os.environ.get(profile.api_key_env)
                if not key:
                    raise MissingCredential("model credential unavailable: " + profile.api_key_env)
                secrets.append(key)
                headers = {"Authorization": "Bearer " + key}
            headers = {**headers, "Content-Type": "application/json",
                       "Accept": "text/event-stream" if body["stream"] else "application/json"}
            request = urllib.request.Request(profile.url, data=wire, method="POST")
            for name, value in headers.items():
                # urllib must never forward credentials to a redirect target.
                if name.lower() in ("authorization", "chatgpt-account-id"):
                    request.add_unredirected_header(name, value)
                else:
                    request.add_header(name, value)
            self._reserve_request()
            reserved = True
            event["dispatch_attempted"] = True
            event["sent_unknown"] = True
            response = urllib.request.urlopen(request, timeout=profile.timeout)
            event.update(request_sent=True, response_received=True, sent_unknown=False)
            with response:
                event["http_status"] = getattr(response, "status", None)
                if body["stream"]:
                    lines = captured_lines(response, event["events"], usage)
                    callback = on_event or (lambda *args, **kwargs: None)
                    raw = read_responses(lines, callback, event["events"])
                    event["response"] = raw
                    usage.update(normalized_usage(raw.get("usage")))
                    result = normalize_response(raw)
                else:
                    raw = json.loads(response.read().decode("utf-8"))
                    event["response"] = raw
                    if isinstance(raw, dict):
                        usage.update(normalized_usage(raw.get("usage")))
                    result = normalize_response(raw)
            finish = result["choices"][0]["finish_reason"]
            event["finish_reason"] = finish
            failed = finish not in ("stop", "tool_calls")
            if failed:
                event["error"] = {"type": "ProviderOutcome", "reason": finish}
            return result
        except BaseException as exc:
            if isinstance(exc, urllib.error.HTTPError):
                event.update(request_sent=True, response_received=True, sent_unknown=False)
                event["http_status"] = exc.code
                reason = "provider HTTP request failed"
                try:
                    error_text = exc.read().decode("utf-8", "replace")
                    try:
                        event["response"] = json.loads(error_text)
                    except json.JSONDecodeError:
                        event["response"] = {"error_body_text": error_text}
                except Exception as read_error:
                    event["response_read_error"] = type(read_error).__name__
                finally:
                    exc.close()
            elif isinstance(exc, (ProviderError, StreamProtocolError, MissingCredential, BudgetExceeded)):
                reason = str(exc)
            else:
                reason = "provider call failed: " + type(exc).__name__
            if (isinstance(exc, socket.gaierror) or
                    isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, socket.gaierror)):
                event["sent_unknown"] = False
            event["error"] = {"type": type(exc).__name__, "reason": reason}
            if isinstance(exc, (ProviderError, StreamProtocolError, MissingCredential, BudgetExceeded)):
                raise
            if not isinstance(exc, Exception):
                raise
            raise ProviderError(reason) from None
        finally:
            event["usage"] = dict(usage)
            event["elapsed_seconds"] = time.monotonic() - started
            # Even a callback cancellation, malformed response or broken stream
            # accounts for observed tokens exactly once. Recorder failure cannot
            # bypass ledger accounting.
            try:
                if reserved:
                    self._record_usage(usage, error=failed)
            finally:
                if self.recorder is not None:
                    self.recorder(redact_record(event, secrets))

    def _persist(self) -> None:
        with _USAGE_LOCK:
            _persist_usage(self.config)

    def reset_budget(self) -> None:
        with _USAGE_LOCK:
            usage = _shared_usage(self.config)
            usage.requests = usage.input_tokens = usage.output_tokens = usage.errors = 0
            _persist_usage(self.config)
            self.usage = usage

    def _reserve_request(self) -> None:
        """Atomically check the shared ledger and reserve one request slot."""
        with _USAGE_LOCK:
            usage = _shared_usage(self.config)
            ceiling = self.effective_max_requests()
            if usage.requests >= ceiling:
                raise BudgetExceeded(f"request budget reached ({usage.requests}/{ceiling})")
            if usage.input_tokens >= self.config.max_input_tokens:
                raise BudgetExceeded(
                    f"input token budget reached ({usage.input_tokens}/{self.config.max_input_tokens})"
                )
            usage.requests += 1
            _persist_usage(self.config)

    def _record_usage(self, usage: dict[str, Any] | None, *, error: bool = False) -> None:
        """Atomically add this call's tokens (or an error) to the shared ledger."""
        with _USAGE_LOCK:
            shared = _shared_usage(self.config)
            data = usage or {}
            shared.input_tokens += int(data.get("prompt_tokens") or 0)
            shared.output_tokens += int(data.get("completion_tokens") or 0)
            if error:
                shared.errors += 1
            _persist_usage(self.config)

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        response_json: bool = False,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> dict[str, Any]:
        return self._profile_chat(messages, max_tokens=max_tokens, temperature=temperature,
                response_json=response_json, model=model, tools=tools, tool_choice=tool_choice)

    def chat_stream(self, messages, *, on_event, max_tokens=2048, temperature=0.2,
                    response_json=False, model=None, tools=None, tool_choice=None):
        """One reserved request; Responses SSE terminal and partial usage required."""
        return self._profile_chat(messages, on_event=on_event, max_tokens=max_tokens,
                temperature=temperature, response_json=response_json, model=model,
                tools=tools, tool_choice=tool_choice)

    @staticmethod
    def content(payload: dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if not choices:
            return ""
        return (choices[0].get("message") or {}).get("content") or ""

    @staticmethod
    def message(payload: dict[str, Any]) -> dict[str, Any]:
        choices = payload.get("choices") or []
        return (choices[0].get("message") or {}) if choices else {}

    @staticmethod
    def tool_calls(payload: dict[str, Any]) -> list[dict[str, Any]]:
        return (DeepSeekClient.message(payload).get("tool_calls") or [])

    def effective_max_requests(self) -> int:
        """Dev cap and trial quota both apply; the larger ceiling is the trial quota."""
        return max(self.config.max_requests, self.config.trial_max_requests)

    def ask_json(self, system: str, user: str, *, max_tokens: int = 2048) -> dict[str, Any]:
        payload = self.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            response_json=True,
        )
        text = self.content(payload).strip()
        if not text:
            usage = payload.get("usage") or {}
            finish = (payload.get("choices") or [{}])[0].get("finish_reason")
            raise RuntimeError(
                f"empty content (finish={finish}, completion_tokens={usage.get('completion_tokens')}); "
                "likely reasoning consumed the token budget"
            )
        return extract_json(text)
