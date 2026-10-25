"""Metered model calls (api.deepseek.com / deepseek-flash only).

Reuses the read-only prototype client. Every request is logged with
kind / elapsed / token usage / success, and the independent ledger is
incremented. Local SQL/tool work is logged separately and is NOT an API call.
"""
from __future__ import annotations

import datetime as dt
import time

from lettaproto.config import Config as PConfig
from lettaproto.deepseek import DeepSeekClient, extract_json

from . import settings as S

__all__ = ["call", "client", "content", "message", "tool_calls", "extract_json", "local_event"]

_CLIENT = None


def client() -> DeepSeekClient:
    global _CLIENT
    if _CLIENT is None:
        c = PConfig()
        c.usage_file = S.USAGE
        c.chat_model = S.MODEL
        c.api_base_url = "https://api.deepseek.com"
        c.max_requests = 1000000
        c.trial_max_requests = 1000000
        c.max_input_tokens = 1000000000
        S.ensure_dirs()
        _CLIENT = DeepSeekClient(c)
    return _CLIENT


def call(kind, msgs, **kw):
    S.guard(kind)
    t0 = time.time()
    ok, usage, err = True, None, None
    try:
        p = client().chat(msgs, **kw)
        usage = p.get("usage") or {}
        S.bump_ledger(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        return p
    except BaseException as exc:  # noqa: BLE001
        ok = False
        err = f"{type(exc).__name__}: {str(exc)[:160]}"
        S.bump_ledger(0, 0, error=True)
        raise
    finally:
        S.log_event({"ts": dt.datetime.now(S.UTC).isoformat(), "type": "api", "kind": kind,
                     "model": S.MODEL, "n_msgs": len(msgs), "tools": bool(kw.get("tools")),
                     "response_json": bool(kw.get("response_json")),
                     "elapsed_s": round(time.time() - t0, 2), "ok": ok,
                     "usage": usage, "error": err})


def local_event(tool, detail):
    S.log_event({"ts": dt.datetime.now(S.UTC).isoformat(), "type": "local", "tool": tool,
                 "detail": detail})


def content(payload):
    return DeepSeekClient.content(payload)


def message(payload):
    return DeepSeekClient.message(payload)


def tool_calls(payload):
    return DeepSeekClient.tool_calls(payload)
