"""Minimal DeepSeek (OpenAI-compatible) client with a hard request/token budget.

The API key is read ONLY from the process environment (injected by the trusted
credential wrapper). It is never written to disk, argv, or logs.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .config import Config
from .util import read_json, write_json


class BudgetExceeded(RuntimeError):
    pass


class MissingCredential(RuntimeError):
    pass


_USAGE_LOCK = threading.Lock()


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
    def __init__(self, config: Config):
        self.config = config
        with _USAGE_LOCK:
            self.usage = Usage(**read_json(config.usage_file, {}) or {})

    def _persist(self) -> None:
        with _USAGE_LOCK:
            write_json(self.config.usage_file, self.usage.to_dict())

    def reset_budget(self) -> None:
        self.usage = Usage()
        self._persist()

    def _check_budget(self) -> None:
        ceiling = self.effective_max_requests()
        if self.usage.requests >= ceiling:
            raise BudgetExceeded(f"request budget reached ({self.usage.requests}/{ceiling})")
        if self.usage.input_tokens >= self.config.max_input_tokens:
            raise BudgetExceeded(
                f"input token budget reached ({self.usage.input_tokens}/{self.config.max_input_tokens})"
            )

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
        key = self.config.api_key
        if not key:
            raise MissingCredential(
                "DEEPSEEK_API_KEY not present in environment; run via the credential wrapper (run.sh)."
            )
        self._check_budget()
        body: dict[str, Any] = {
            "model": model or self.config.chat_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if response_json:
            body["response_format"] = {"type": "json_object"}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice or "auto"
        url = self.config.api_base_url.rstrip("/") + "/v1/chat/completions"
        request = urllib.request.Request(
            url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        self.usage.requests += 1
        try:
            with urllib.request.urlopen(request, timeout=180) as resp:
                payload = json.loads(resp.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as err:
            self.usage.errors += 1
            detail = err.read().decode("utf-8", "replace")[:500]
            self._persist()
            raise RuntimeError(f"HTTP {err.code}: {detail}") from err
        except Exception as exc:  # noqa: BLE001
            self.usage.errors += 1
            self._persist()
            raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc
        usage = payload.get("usage") or {}
        self.usage.input_tokens += int(usage.get("prompt_tokens") or 0)
        self.usage.output_tokens += int(usage.get("completion_tokens") or 0)
        self._persist()
        return payload

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
