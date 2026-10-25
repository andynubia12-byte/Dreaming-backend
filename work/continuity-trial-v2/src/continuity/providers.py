"""One Responses protocol adapter for every model and agent role.

Lineage: llmtest/server.mjs buildResponsesPayload (596), makeRequestDebug (630),
mapCodexUsage (484), Responses event switch (749), responsesUrl (209).
Source SHA-256: d67a84965dc1b0af45b4ffbbed7c163692bd0765ae0783a46eac12555537df8f.
The historical provider audit is not a runtime dependency.
Ported field/event semantics, not its role filtering, EOF success, credential
discovery/refresh, or independent tool loop. No network or auth reads on creation.

Recorder contract: one ``type=provider_call`` event per call (also on early
failure), with role/model/api_format/url, request (final JSON body), exact wire
request_body_sha256, dispatch_attempted, response_received, request_sent,
sent_unknown, response, events (SSE), usage, elapsed_seconds, http_status, error,
redactions, record_changed and replay_limited. request_sent requires an HTTP
response; a pre-response network failure is unknown unless DNS failed.
No headers are recorded.
Reasoning remains intact in memory/on the protocol; persisted plaintext reasoning
is replaced with a marker and path/len/sha256 metadata. Credential redactions
also mark replay limited, with path/len but no original-value hash. Therefore a
redacted record is not an exact replay artifact. Encrypted reasoning stays opaque.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from urllib.parse import urlsplit, urlunsplit

from .deepseek import DeepSeekClient
from .provider_stream import StreamProtocolError


class ProviderError(RuntimeError):
    """A safe local diagnostic, never a provider-supplied error message."""


def create_client(config, profile: dict, *, recorder=None, role="main"):
    return DeepSeekClient(config, profile=profile, recorder=recorder, role=role)


def endpoint(base_url, api_format):
    if api_format != "responses":
        raise ValueError("only the Responses API is supported")
    if not isinstance(base_url, str):
        raise ValueError("base_url must be an http(s) URL without credentials/query/fragment")
    parts = urlsplit(base_url)
    if (parts.scheme not in ("http", "https") or not parts.hostname
            or parts.username is not None or parts.password is not None
            or "?" in base_url or "#" in base_url or any(c.isspace() for c in base_url)):
        raise ValueError("base_url must be an http(s) URL without credentials/query/fragment")
    path = parts.path.rstrip("/")
    if path.endswith("/chat/completions"):
        raise ValueError("Chat Completions endpoints are not supported; configure a Responses endpoint")
    if not path.endswith("/responses"):
        path += "/responses"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def migrate_legacy_profile(data):
    """Explicit metadata migration for saved pre-Responses experiments only.

    New API input is always validated by Profile, never passed through here.
    Do not rewrite historical wire artifacts or retry a request on another API.
    Parameters without a Responses equivalent fail instead of being discarded.
    """
    result = copy.deepcopy(data)
    if result.get("api_format") != "chat_completions":
        return result
    result["api_format"] = "responses"
    base = result.get("base_url", "").rstrip("/")
    if base.endswith("/chat/completions"):
        result["base_url"] = base[:-len("/chat/completions")]
    p = result.setdefault("parameters", {})
    for old in ("max_tokens", "max_completion_tokens"):
        if old in p:
            if "max_output_tokens" in p:
                raise ValueError("conflicting legacy token limits; reconfigure this saved model")
            p["max_output_tokens"] = p.pop(old)
    for where in (result, p):
        if "token_limit_field" in where:
            if where["token_limit_field"] not in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
                raise ValueError("invalid legacy token limit field")
            where["token_limit_field"] = "max_output_tokens"
    effort = p.pop("reasoning_effort", None)
    thinking = p.pop("thinking", None)
    if thinking is not None:
        if thinking not in ({"type": "enabled"}, {"type": "disabled"}):
            raise ValueError("invalid legacy thinking option")
        if thinking == {"type": "disabled"}:
            if effort not in (None, "none"):
                raise ValueError("conflicting legacy reasoning options")
            effort = "none"
        elif effort == "none":
            raise ValueError("conflicting legacy reasoning options")
        elif effort is None:
            effort = "high"
    if effort is not None:
        if "reasoning" in p:
            raise ValueError("conflicting legacy reasoning options")
        p["reasoning"] = {"effort": effort}
    if "response_format" in p:
        if "text" in p:
            raise ValueError("conflicting legacy text options")
        fmt = p.pop("response_format")
        if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
            fmt = {"type": "json_schema", **fmt.get("json_schema", {})}
        p["text"] = {"format": fmt}
    return result


def _number(value, name, low, high):
    if (isinstance(value, bool) or not isinstance(value, (float, int))
            or not math.isfinite(value) or not low <= value <= high):
        raise ValueError(f"invalid {name}")


class Profile:
    def __init__(self, data):
        if not isinstance(data, dict):
            raise ValueError("profile must be an object")
        allowed = {"id", "model", "base_url", "api_format", "api_key_env", "parameters",
                   "timeout_seconds", "token_limit_field", "auth"}
        if set(data) - allowed:
            raise ValueError("unsupported profile field")
        for field in ("id", "model", "base_url", "api_format"):
            if not isinstance(data.get(field), str) or not data[field].strip():
                raise ValueError(f"profile requires {field}")
        self.id, self.model, self.api_format = data["id"], data["model"], data["api_format"]
        if self.api_format != "responses":
            raise ValueError("only the Responses API is supported")
        self.url = endpoint(data["base_url"], self.api_format)
        self.auth = data.get("auth", "api_key")
        if self.auth not in ("api_key", "codex_subscription"):
            raise ValueError("unsupported auth")
        if self.auth == "codex_subscription":
            from .codex_credentials import validate_codex_base_url
            if self.api_format != "responses":
                raise ValueError("codex_subscription requires the fixed Codex Responses endpoint")
            validate_codex_base_url(data["base_url"])
        self.api_key_env = data.get("api_key_env")
        if self.auth == "api_key" and (not isinstance(self.api_key_env, str)
                or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env)):
            raise ValueError("profile requires an api_key_env name")
        self.timeout = data.get("timeout_seconds", 180)
        _number(self.timeout, "timeout_seconds", 0.001, 3600)
        self.parameters = copy.deepcopy(data.get("parameters", {}))
        if not isinstance(self.parameters, dict):
            raise ValueError("parameters must be an object")
        p = self.parameters
        common = {"temperature", "top_p", "service_tier", "parallel_tool_calls"}
        responses = {"max_output_tokens", "reasoning", "text", "max_tool_calls"}
        if set(p) - (common | responses | {"token_limit_field"}):
            raise ValueError("unsupported or protected generation parameter")
        parameter_field = p.pop("token_limit_field", None)
        self.token_field = data.get("token_limit_field", parameter_field)
        if parameter_field and self.token_field != parameter_field:
            raise ValueError("conflicting token_limit_field values")
        valid_fields = {"max_output_tokens"}
        if self.token_field is not None and self.token_field not in valid_fields:
            raise ValueError("invalid token_limit_field for api_format")
        caps = [k for k in valid_fields if k in p]
        if len(caps) > 1 or (caps and self.token_field and caps[0] != self.token_field):
            raise ValueError("conflicting token limit fields")
        self.token_field = self.token_field or "max_output_tokens"
        for name in caps:
            if type(p[name]) is not int or p[name] <= 0:
                raise ValueError("token limit must be a positive integer")
        for name, bounds in {"temperature": (0, 2), "top_p": (0, 1)}.items():
            if name in p and not (name == "temperature" and p[name] is None):
                _number(p[name], name, *bounds)
        if "service_tier" in p and p["service_tier"] not in ("default", "flex"):
            raise ValueError("only default/flex service tiers are allowed; Fast/priority is forbidden")
        if "parallel_tool_calls" in p and type(p["parallel_tool_calls"]) is not bool:
            raise ValueError("parallel_tool_calls must be boolean")
        if "max_tool_calls" in p and (type(p["max_tool_calls"]) is not int or p["max_tool_calls"] <= 0):
            raise ValueError("max_tool_calls must be a positive integer")
        efforts = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
        if "reasoning" in p:
            r = p["reasoning"]
            if (not isinstance(r, dict) or set(r) - {"effort", "summary"}
                    or "effort" in r and r["effort"] not in efforts
                    or "summary" in r and r["summary"] not in ("auto", "concise", "detailed")):
                raise ValueError("invalid reasoning options")
        for name in ("text",):
            if name in p:
                value = p[name]
                if not isinstance(value, dict):
                    raise ValueError(f"invalid {name}")
                fmt = value.get("format")
                if fmt is not None and (not isinstance(fmt, dict) or fmt.get("type") not in
                                        ("text", "json_object", "json_schema")):
                    raise ValueError(f"invalid {name} format")
                if (set(value) - {"format", "verbosity"} or
                        "verbosity" in value and value["verbosity"] not in ("low", "medium", "high")):
                    raise ValueError("unsupported text option")
        json.dumps(p, allow_nan=False)

    def body(self, client, messages, *, max_tokens, temperature, response_json, model,
             tools, tool_choice, stream):
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        # Existing main/retrieval loops pass their configured default explicitly.
        route_default = (getattr(client.config, "main_chat_model", None) if client.role == "main"
                         else client.config.chat_model)
        selected = self.model if (not model or
            getattr(client, "_profile_supplied", True) and model == route_default) else model
        p = copy.deepcopy(self.parameters)
        cap = min(max_tokens, p.pop(self.token_field, max_tokens))
        temp = p.pop("temperature", temperature)
        if temp is not None:
            _number(temp, "temperature", 0, 2)
        # Optional service tiers belong to explicit profile parameters. Some
        # Responses providers reject an unsolicited "default" value.
        body = {"model": selected, "input": response_input(messages), "stream": stream,
                "max_output_tokens": cap, "store": False,
                "include": ["reasoning.encrypted_content"]}
        if self.auth == "codex_subscription":
            body["stream"] = True
            body["instructions"] = ""
        if temp is not None:
            body["temperature"] = temp
        if response_json:
            body["text"] = {"format": {"type": "json_object"}}
        if tools:
            body["tools"] = []
            for tool in tools:
                if tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
                    raise ValueError("only function tools are supported")
                body["tools"].append({"type": "function", "strict": False,
                                      **copy.deepcopy(tool["function"])})
            body["tool_choice"] = tool_choice or "auto"
        if "text" in p:
            body["text"] = {**body.get("text", {}), **p.pop("text")}
        body.update(p)
        return body


def response_input(messages):
    items = []
    for message in messages:
        role = message["role"]
        if role == "assistant" and "_response_items" in message:
            if not isinstance(message["_response_items"], list):
                raise ValueError("_response_items must be a list")
            items.extend(copy.deepcopy(message["_response_items"]))
        elif role == "tool":
            items.append({"type": "function_call_output", "call_id": message["tool_call_id"],
                          "output": message.get("content") or ""})
        else:
            # Preserve actual roles, exact strings, and structured content.
            if message.get("content") is not None:
                items.append({"role": role, "content": copy.deepcopy(message["content"])})
            for call in message.get("tool_calls") or []:
                items.append({"type": "function_call", "call_id": call["id"],
                              "name": call["function"]["name"],
                              "arguments": call["function"]["arguments"]})
    return items


def normalized_usage(usage):
    usage = dict(usage or {})
    if "input_tokens" in usage:
        usage["prompt_tokens"] = usage["input_tokens"]
    if "output_tokens" in usage:
        usage["completion_tokens"] = usage["output_tokens"]
    return usage


def normalize_response(payload):
    if not isinstance(payload, dict):
        raise ProviderError("invalid provider response")
    status = payload.get("status")
    if status not in ("completed", "incomplete") or payload.get("error"):
        raise ProviderError("provider response failed or has no terminal status")
    output = payload.get("output")
    if not isinstance(output, list):
        raise ProviderError("provider response missing output items")
    texts, native_calls, refusals = [], [], []
    for item in output:
        if item.get("type") == "message":
            for part in item.get("content") or []:
                if part.get("type") == "output_text":
                    texts.append(part["text"])
                elif part.get("type") == "refusal":
                    refusals.append(part.get("refusal") or "refused")
        elif item.get("type") == "function_call":
            native_calls.append(item)
    if status == "incomplete":
        finish = ("length" if (payload.get("incomplete_details") or {}).get("reason") ==
                  "max_output_tokens" else "incomplete")
    elif refusals:
        finish = "content_filter"
    else:
        finish = "tool_calls" if native_calls else "stop"
    if finish == "stop" and not "".join(texts).strip():
        raise ProviderError("provider returned no visible answer or function calls")
    message = {"role": "assistant", "content": "".join(texts),
               "_response_items": copy.deepcopy(output)}
    if finish == "tool_calls":
        calls = []
        for item in native_calls:
            if (not item.get("call_id") or not item.get("name")
                    or item.get("status", "completed") != "completed"):
                raise ProviderError("incomplete function call")
            try:
                if not isinstance(json.loads(item["arguments"]), dict):
                    raise ValueError()
            except (ValueError, KeyError, TypeError) as exc:
                raise ProviderError("incomplete function arguments") from exc
            calls.append({"id": item["call_id"], "type": "function",
                          "function": {"name": item["name"], "arguments": item["arguments"]}})
        if len({c["id"] for c in calls}) != len(calls):
            raise ProviderError("duplicate function call ids")
        message["tool_calls"] = calls
    if refusals:
        message["refusal"] = "".join(refusals)
    return {"choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": normalized_usage(payload.get("usage")), "id": payload.get("id"),
            "status": status}


def captured_lines(response, events, usage):
    """Capture complete SSE data frames, preserving partial streams on failure.

    Yield unchanged lines to the Responses parser. Both LF and CRLF are legal;
    malformed data is rejected rather than silently discarded as in the source.
    """
    data = []

    def consume():
        raw = "\n".join(data)
        data.clear()
        if raw == "[DONE]":
            events.append({"type": "done"})
            return
        try:
            event = json.loads(raw)
            if not isinstance(event, dict):
                raise ValueError()
        except ValueError as exc:
            raise StreamProtocolError("invalid provider SSE data") from exc
        events.append(event)
        observed = event.get("usage") or (event.get("response") or {}).get("usage")
        if observed:
            usage.update(normalized_usage(observed))

    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
        elif not line and data:
            consume()
        yield raw
    if data:
        consume()


def read_responses(lines, on_event, events):
    """True SSE callbacks; terminal native response is authoritative for replay.

    Native terminal responses contain all final output items, including argument
    deltas, multiple tools, refusals, and encrypted reasoning. Never reconstruct
    a lossy tool loop from text deltas or accept EOF as a completed response.
    """
    seen = 0
    terminal = None
    reasoning_chars = 0
    refused = False
    items, arguments = {}, {}

    def consume(event):
        nonlocal terminal, reasoning_chars, refused
        kind = event.get("type", "")
        if not isinstance(kind, str) or not kind:
            raise StreamProtocolError("invalid provider event type")
        if kind == "done" and terminal is None:
            raise StreamProtocolError("provider stream missing terminal response")
        if terminal is not None and kind != "done":
            raise StreamProtocolError("provider event after terminal response")
        if kind == "response.output_text.delta":
            if not isinstance(event.get("delta"), str):
                raise StreamProtocolError("invalid output text delta")
            on_event("content", text=event["delta"])
        elif kind in ("response.refusal.delta", "response.refusal.done"):
            refused = True
        elif kind.startswith("response.reasoning") and kind.endswith(".delta"):
            reasoning_chars += len(event.get("delta") or "")
            on_event("thinking", chars=reasoning_chars)
        elif kind in ("response.output_item.added", "response.output_item.done",
                      "response.function_call_arguments.delta", "response.function_call_arguments.done"):
            index = event.get("output_index")
            if type(index) is not int or index < 0:
                raise StreamProtocolError("invalid output item index")
            if kind.startswith("response.output_item"):
                item = event.get("item")
                if not isinstance(item, dict):
                    raise StreamProtocolError("invalid output item")
                items[index] = (item, kind.endswith(".done"))
            elif kind.endswith(".done"):
                if not isinstance(event.get("arguments"), str):
                    raise StreamProtocolError("invalid function arguments")
                arguments[index] = event["arguments"]
            elif not isinstance(event.get("delta"), str):
                raise StreamProtocolError("invalid function arguments delta")
        elif kind in ("response.completed", "response.incomplete", "response.failed"):
            terminal = event.get("response")
            if not isinstance(terminal, dict) or terminal.get("status") != kind.split(".")[-1]:
                raise StreamProtocolError("invalid provider terminal response")
            # Completed streams cannot drop observed native items or silently
            # lose a second tool. Incomplete/failed responses retain partials in
            # the recorder and are never classified as successful.
            if terminal["status"] == "completed":
                output = terminal.get("output") or []
                if refused and not any(p.get("type") == "refusal" for item in output
                                       for p in item.get("content") or []):
                    raise StreamProtocolError("terminal response lost refusal")
                for index, (item, done) in items.items():
                    if (index >= len(output) or done and output[index] != item
                            or any(output[index].get(k) != item[k] for k in ("id", "type", "call_id", "name")
                                   if k in item)):
                        raise StreamProtocolError("terminal response lost or changed output items")
                for index, args in arguments.items():
                    if index >= len(output) or output[index].get("arguments") != args:
                        raise StreamProtocolError("terminal response lost function arguments")
        elif kind in ("error", "response.error"):
            raise StreamProtocolError("provider error event")

    for _line in lines:
        while seen < len(events):
            consume(events[seen])
            seen += 1
    # Final frame without a blank delimiter was captured at EOF.
    for event in events[seen:]:
        consume(event)
    if terminal is None:
        raise StreamProtocolError("provider stream missing terminal response")
    return terminal


def redact_record(event, secrets=()):
    """Detached, credential-free persisted view, without mutating protocol data."""
    redactions = []

    def hidden(value, path):
        raw = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        metadata = {"path": path, "len": len(raw), "kind": "reasoning"}
        if any(secret and secret in raw for secret in secrets):
            metadata["sha256_omitted"] = "contains credential"
        else:
            metadata["sha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        redactions.append(metadata)
        return "[REDACTED_REASONING]"

    def credential(value, path, replacement="[REDACTED_CREDENTIAL]"):
        raw = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        redactions.append({"path": path, "len": len(raw), "kind": "credential",
                           "sha256_omitted": "credential"})
        return replacement

    def visit(value, path):
        if isinstance(value, dict):
            reasoning = value.get("type") in ("reasoning", "reasoning_text", "summary_text")
            reasoning_event = str(value.get("type", "")).startswith("response.reasoning")
            result = {}
            for key, item in value.items():
                child = path + "/" + str(key).replace("~", "~0").replace("/", "~1")
                if (key in ("reasoning_content", "reasoning_text") or
                        reasoning and key not in ("type", "id", "status", "encrypted_content") or
                        reasoning_event and key in ("delta", "text", "content", "summary", "part")):
                    result[key] = hidden(item, child)
                elif str(key).lower() in ("authorization", "api_key", "access_token", "refresh_token"):
                    result[key] = credential(item, child)
                else:
                    result[key] = visit(item, child)
            return result
        if isinstance(value, list):
            return [visit(item, path + "/" + str(i)) for i, item in enumerate(value)]
        if isinstance(value, str):
            original = value
            for secret in secrets:
                if secret:
                    value = value.replace(secret, "[REDACTED_CREDENTIAL]")
            if value != original:
                return credential(original, path, value)
        return value

    result = visit(event, "")
    result["redactions"] = redactions
    result["record_changed"] = bool(redactions)
    result["replay_limited"] = bool(redactions)
    if redactions:
        result["replay_limitation"] = "Redacted fields differ from wire; exact replay requires the unredacted in-memory protocol items."
    return result
