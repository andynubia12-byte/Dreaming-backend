"""Offline exact-wire Responses-only regression and the shared budget.

Source fixture semantics: llmtest/server.mjs (SHA-256
d67a84965dc1b0af45b4ffbbed7c163692bd0765ae0783a46eac12555537df8f),
buildResponsesPayload/mapCodexUsage/Responses switch. CRLF/EOF/partial-loss
regressions come from the 2026-10-21 source audit. No real credentials/network.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from continuity.config import Config
from continuity.deepseek import DeepSeekClient, BudgetExceeded, MissingCredential
from continuity.providers import create_client, endpoint, ProviderError, redact_record
from continuity.provider_stream import StreamProtocolError


SECRET = "offline-fixture-secret-value"
MESSAGES = [{"role": "system", "content": "system\nrecent stays here\n "},
            {"role": "developer", "content": "developer"},
            {"role": "user", "content": " question \n"}]
TOOLS = [{"type": "function", "function": {"name": "lookup", "description": "search",
          "parameters": {"type": "object", "properties": {"query": {"type": "string"},
                         "optional": {"type": "integer"}}, "required": ["query"]}}}]


def answer(text="visible", **extra):
    return {"type": "message", "id": "msg1", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}], **extra}


def function(call_id="call_a", arguments='{"query":"x"}'):
    return {"type": "function_call", "id": "fc_" + call_id, "call_id": call_id,
            "name": "lookup", "arguments": arguments, "status": "completed"}


def native(output=None, status="completed", **extra):
    return {"id": "resp1", "status": status, "output": [answer()] if output is None else output,
            "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10,
                      "output_tokens_details": {"reasoning_tokens": 2}}, **extra}


def sse(*events, newline="\r\n"):
    return (":" + newline + newline + "".join(
        "data: " + (e if isinstance(e, str) else json.dumps(e, ensure_ascii=False))
        + newline + newline for e in events)).encode()


class Response(io.BytesIO):
    status = 200


class Endpoint:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.bodies = []
        self.timeouts = []

    def __call__(self, request, *, timeout):
        self.requests.append(request)
        self.bodies.append(json.loads(request.data))
        self.timeouts.append(timeout)
        if not self.responses:
            raise AssertionError("unexpected extra provider request")
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return Response(value if isinstance(value, bytes) else json.dumps(value).encode())


class ProvidersTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Config(usage_file=Path(self.tmp.name) / "usage.json")
        self.events = []
        self.env = patch.dict(os.environ, {"PROVIDERS_OFFLINE_KEY": SECRET,
                                          "DEEPSEEK_API_KEY": SECRET})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def profile(self, fmt="responses", **extra):
        return {"id": "fixture", "model": "fixture-model", "api_format": fmt,
                "base_url": "https://fixture.invalid/v1", "api_key_env": "PROVIDERS_OFFLINE_KEY",
                "parameters": {"temperature": None}, **extra}

    def client(self, fmt="responses", **extra):
        return create_client(self.cfg, self.profile(fmt, **extra), recorder=self.events.append)

    def wire(self, endpoint_):
        return patch("continuity.deepseek.urllib.request.urlopen", endpoint_)

    def test_responses_exact_wire_roles_tools_and_cap(self):
        endpoint_ = Endpoint(native())
        client = self.client(parameters={"temperature": None, "max_output_tokens": 1000,
                                         "reasoning": {"effort": "high"}})
        with self.wire(endpoint_):
            result = client.chat(MESSAGES, max_tokens=32000, temperature=0.6, tools=TOOLS)
        body = endpoint_.bodies[0]
        self.assertEqual(body, {"model": "fixture-model", "input": MESSAGES, "stream": False,
                         "max_output_tokens": 1000, "store": False,
                         "include": ["reasoning.encrypted_content"],
                         "reasoning": {"effort": "high"},
                         "tools": [{"type": "function", **TOOLS[0]["function"], "strict": False}],
                         "tool_choice": "auto"})
        self.assertEqual(client.content(result), "visible")
        self.assertEqual(client.usage.to_dict(), {"requests": 1, "input_tokens": 7,
                                                "output_tokens": 3, "errors": 0})
        event = self.events[0]
        self.assertEqual(event["request"], body)
        self.assertEqual(event["request_body_sha256"], hashlib.sha256(endpoint_.requests[0].data).hexdigest())
        self.assertEqual(event["response"], native())
        self.assertEqual(event["usage"]["output_tokens_details"], {"reasoning_tokens": 2})
        self.assertEqual(event["http_status"], 200)
        self.assertGreaterEqual(event["elapsed_seconds"], 0)

    def test_kimi_request_has_no_implicit_service_tier(self):
        endpoint_ = Endpoint(native())
        client = self.client(model="kimi-k2.7", base_url="https://api.moonshot.cn/v1",
            parameters={"reasoning": {"effort": "medium"}, "temperature": None})
        with self.wire(endpoint_):
            client.chat(MESSAGES)
        self.assertEqual(endpoint_.requests[0].full_url, "https://api.moonshot.cn/v1/responses")
        self.assertNotIn("service_tier", endpoint_.bodies[0])
        self.assertEqual(endpoint_.bodies[0]["reasoning"], {"effort": "medium"})
        self.assertEqual(endpoint_.bodies[0]["input"], MESSAGES)
        self.assertEqual(len(endpoint_.requests), 1)

    def test_service_tier_is_sent_only_when_explicitly_configured(self):
        endpoint_ = Endpoint(native())
        with self.wire(endpoint_):
            self.client(parameters={"service_tier": "default"}).chat(MESSAGES)
        self.assertEqual(endpoint_.bodies[0]["service_tier"], "default")

    def test_explicit_tool_strict_values_are_preserved(self):
        for strict in (True, False, "absent"):
            with self.subTest(strict=strict):
                tools = json.loads(json.dumps(TOOLS))
                if strict != "absent":
                    tools[0]["function"]["strict"] = strict
                original = json.loads(json.dumps(tools))
                endpoint_ = Endpoint(native())
                with self.wire(endpoint_):
                    self.client().chat(MESSAGES, tools=tools)
                expected = False if strict == "absent" else strict
                self.assertIs(endpoint_.bodies[0]["tools"][0]["strict"], expected)
                self.assertEqual(tools, original)

    def test_multiple_native_tool_turns_replay_every_item_in_order(self):
        raw_reason = "private reasoning never persisted"
        first = [{"type": "reasoning", "id": "rs1", "encrypted_content": "opaque==",
                  "content": [{"type": "reasoning_text", "text": raw_reason}],
                  "summary": [{"type": "summary_text", "text": "private summary"}]},
                 answer("before tools"), function(), function("call_b")]
        second = [{"type": "reasoning", "id": "rs2", "encrypted_content": "opaque2==",
                   "summary": []}, answer("again"), function("call_c")]
        endpoint_ = Endpoint(native(first), native(second), native())
        client = self.client()
        messages = list(MESSAGES)
        with self.wire(endpoint_):
            for expected in (first, second):
                result = client.chat(messages, tools=TOOLS)
                msg = client.message(result)
                self.assertEqual(msg["_response_items"], expected)
                messages.append(msg)
                for call in client.tool_calls(result):
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": "found"})
            client.chat(messages)
        expected = MESSAGES + first + [
            {"type": "function_call_output", "call_id": "call_a", "output": "found"},
            {"type": "function_call_output", "call_id": "call_b", "output": "found"}] + second + [
            {"type": "function_call_output", "call_id": "call_c", "output": "found"}]
        self.assertEqual(endpoint_.bodies[2]["input"], expected)
        persisted = json.dumps(self.events)
        self.assertNotIn(raw_reason, persisted)
        self.assertNotIn("private summary", persisted)
        self.assertIn("opaque==", persisted)
        self.assertTrue(self.events[2]["replay_limited"])
        redaction = next(r for r in self.events[0]["redactions"] if r["path"].endswith("/content"))
        self.assertEqual(redaction["sha256"], hashlib.sha256(json.dumps(first[0]["content"],
                         ensure_ascii=False).encode()).hexdigest())

    def test_chat_completions_profiles_are_rejected_before_transport(self):
        from continuity.providers import Profile
        profile = self.profile("chat_completions")
        with self.wire(Endpoint()) as transport:
            with self.assertRaises(ValueError):
                Profile(profile)
            with self.assertRaises(ValueError):
                create_client(self.cfg, profile)
        self.assertEqual(self.events, [])

    def test_unprofiled_deepseek_defaults_to_responses(self):
        endpoint_ = Endpoint(native())
        with self.wire(endpoint_):
            result = DeepSeekClient(self.cfg).chat(MESSAGES, tools=TOOLS)
        self.assertEqual(endpoint_.requests[0].full_url, endpoint(self.cfg.api_base_url, "responses"))
        self.assertEqual(endpoint_.bodies[0], {"model": self.cfg.chat_model, "input": MESSAGES,
            "max_output_tokens": 2048, "temperature": 0.2, "stream": False,
            "store": False, "include": ["reasoning.encrypted_content"],
            "tools": [{"type": "function", **TOOLS[0]["function"], "strict": False}],
            "tool_choice": "auto", "reasoning": {"effort": "high"}})
        self.assertEqual(DeepSeekClient.content(result), "visible")

    def test_legacy_profile_migration_is_explicit_and_does_not_mutate_input(self):
        from continuity.providers import Profile, migrate_legacy_profile
        legacy = self.profile("chat_completions", base_url="https://fixture.invalid/v1/chat/completions",
            token_limit_field="max_completion_tokens", parameters={
                "max_completion_tokens": 900, "reasoning_effort": "none", "temperature": None})
        original = json.loads(json.dumps(legacy))
        with self.assertRaises(ValueError):
            Profile(legacy)
        migrated = migrate_legacy_profile(legacy)
        self.assertEqual(legacy, original)
        self.assertEqual(migrated["api_format"], "responses")
        self.assertEqual(migrated["parameters"]["reasoning"], {"effort": "none"})
        endpoint_ = Endpoint(native())
        with self.wire(endpoint_):
            create_client(self.cfg, migrated).chat(MESSAGES)
        self.assertEqual(endpoint_.requests[0].full_url, "https://fixture.invalid/v1/responses")
        self.assertEqual(endpoint_.bodies[0]["max_output_tokens"], 900)
        self.assertEqual(endpoint_.bodies[0]["reasoning"], {"effort": "none"})
        self.assertNotIn("messages", endpoint_.bodies[0])

    def test_stream_cannot_lose_observed_item_failure_status(self):
        for item in (function(), answer()):
            observed = {**item, "status": "incomplete"}
            for terminal_item in (item, {k: v for k, v in item.items() if k != "status"}):
                with self.subTest(item=item["type"], terminal=terminal_item.get("status")):
                    wire = sse({"type": "response.output_item.done", "output_index": 0,
                                "item": observed},
                               {"type": "response.completed", "response": native([terminal_item])})
                    with self.wire(Endpoint(wire)), self.assertRaises(StreamProtocolError):
                        self.client().chat_stream(MESSAGES, on_event=lambda *a, **kw: None)

    def test_unprofiled_deepseek_honors_explicit_model_in_both_modes(self):
        for streaming in (False, True):
            for model in (self.cfg.main_chat_model, "explicit-model"):
                with self.subTest(streaming=streaming, model=model):
                    raw = native()
                    endpoint_ = Endpoint(sse({"type": "response.completed", "response": raw}) if streaming else raw)
                    client = DeepSeekClient(self.cfg)
                    with self.wire(endpoint_):
                        if streaming:
                            client.chat_stream(MESSAGES, model=model, on_event=lambda *a, **kw: None)
                        else:
                            client.chat(MESSAGES, model=model)
                    self.assertEqual(endpoint_.bodies[0]["model"], model)
                    self.assertTrue(endpoint_.requests[0].full_url.endswith("/responses"))

    def test_responses_multitool_timeout_and_explicit_token_field(self):
        calls = [function("a", "{}"), function("b", "{}")]
        endpoint_ = Endpoint(native([answer("before"), *calls]), native())
        client = self.client(token_limit_field="max_output_tokens", timeout_seconds=25,
                             parameters={"temperature": None})
        with self.wire(endpoint_):
            first = client.chat(MESSAGES, tools=TOOLS, max_tokens=900)
            continuation = MESSAGES + [client.message(first)] + [
                {"role": "tool", "tool_call_id": call["call_id"], "content": "result"} for call in calls]
            client.chat(continuation, tools=TOOLS)
        self.assertEqual(endpoint_.bodies[1]["input"], MESSAGES + [answer("before"), *calls] + [
            {"type": "function_call_output", "call_id": name, "output": "result"} for name in ("a", "b")])
        self.assertEqual(endpoint_.bodies[0]["max_output_tokens"], 900)
        self.assertEqual(endpoint_.timeouts, [25, 25])

    def test_responses_malformed_function_call_is_not_success(self):
        client = self.client()
        with self.wire(Endpoint(native([function(arguments="{")]))), self.assertRaises(ProviderError):
            client.chat(MESSAGES)
        self.assertEqual(client.usage.errors, 1)

    def test_model_routing_and_explicit_override(self):
        main = self.client()
        retrieval = create_client(self.cfg, self.profile(model="retrieval-model"),
                                  role="retrieval", recorder=self.events.append)
        endpoint_ = Endpoint(native(), native(), native())
        with self.wire(endpoint_):
            main.chat(MESSAGES, model=self.cfg.main_chat_model)
            retrieval.chat(MESSAGES, model=self.cfg.chat_model)
            main.chat(MESSAGES, model="explicit-model")
        self.assertEqual([b["model"] for b in endpoint_.bodies],
                         ["fixture-model", "retrieval-model", "explicit-model"])
        self.assertIs(main.usage, retrieval.usage)
        self.assertEqual(main.usage.requests, 3)
        self.assertEqual(main.usage.input_tokens, 21)
        self.assertEqual([e["role"] for e in self.events], ["main", "retrieval", "main"])

    def test_both_clients_share_request_cap(self):
        self.cfg.max_requests = self.cfg.trial_max_requests = 1
        main, retrieval = self.client(), self.client()
        endpoint_ = Endpoint(native())
        with self.wire(endpoint_):
            main.chat(MESSAGES)
            with self.assertRaises(BudgetExceeded):
                retrieval.chat(MESSAGES)
        self.assertEqual(main.usage.to_dict(), {"requests": 1, "input_tokens": 7,
                                              "output_tokens": 3, "errors": 0})
        self.assertFalse(self.events[-1]["request_sent"])
        self.assertIsNotNone(self.events[-1]["request"])

    def test_endpoint_joining_and_legacy_format_rejection(self):
        for base, suffix in [
            ("https://fixture.invalid", "/responses"),
            ("https://fixture.invalid/v1/", "/v1/responses"),
            ("https://fixture.invalid/responses", "/responses"),
        ]:
            self.assertEqual(endpoint(base, "responses"), "https://fixture.invalid" + suffix)
            with self.assertRaises(ValueError):
                endpoint(base, "chat_completions")
        with self.assertRaises(ValueError):
            endpoint("https://fixture.invalid/v1/chat/completions", "responses")

    def test_reject_credential_urls_without_echoing_them(self):
        for url in ("https://user:secret@host/v1", "https://host/?key=secret", "https://host/#secret",
                    "https://host/?", "ftp://host", "https://host/#"):
            with self.subTest(url=url), self.assertRaises(ValueError) as caught:
                self.client(base_url=url)
            self.assertNotIn("secret", str(caught.exception))

    def test_parameters_protected_and_validated(self):
        for name in ("model", "messages", "input", "tools", "stream", "instructions", "store",
                     "truncation", "previous_response_id", "conversation", "include", "unknown"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.client(parameters={name: "override"})
        for name, value in [("temperature", 3), ("temperature", True), ("top_p", -1),
                            ("reasoning", {"effort": "guessed"}), ("max_output_tokens", None),
                            ("max_output_tokens", -1), ("parallel_tool_calls", "false"),
                            ("service_tier", "fast"), ("service_tier", "priority"),
                            ("service_tier", "ultrafast"), ("service_tier", "auto")]:
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                self.client(parameters={name: value})
        with self.assertRaises(ValueError):
            self.client(parameters={"max_tokens": 1, "max_output_tokens": 2})
        with self.assertRaises(ValueError):
            self.client(token_limit_field="max_tokens",
                        parameters={"token_limit_field": "max_output_tokens"})

    def test_explicit_generation_options_preserved_and_json_mode_merged(self):
        endpoint_ = Endpoint(native())
        client = self.client(parameters={"temperature": 0.4, "top_p": 0.9,
                             "parallel_tool_calls": False, "text": {"verbosity": "low"}})
        with self.wire(endpoint_):
            client.chat(MESSAGES, temperature=0.8, response_json=True, tools=TOOLS)
        body = endpoint_.bodies[0]
        self.assertEqual(body["temperature"], 0.4)
        self.assertEqual(body["top_p"], 0.9)
        self.assertIs(body["parallel_tool_calls"], False)
        self.assertEqual(body["text"], {"format": {"type": "json_object"}, "verbosity": "low"})

    def test_http_error_has_original_body_safe_status_and_no_retry(self):
        error = urllib.error.HTTPError("https://ignored.invalid/?key=" + SECRET, 400,
                    "echo " + SECRET, {}, io.BytesIO(("echo " + SECRET).encode()))
        endpoint_ = Endpoint(error)
        client = self.client(parameters={"temperature": None, "reasoning": {"effort": "high"}})
        with self.wire(endpoint_), self.assertRaises(ProviderError):
            client.chat(MESSAGES)
        event = self.events[0]
        self.assertEqual(event["http_status"], 400)
        self.assertEqual(event["request"]["reasoning"], {"effort": "high"})
        self.assertEqual(client.usage.errors, 1)
        self.assertEqual(len(endpoint_.requests), 1)
        self.assertNotIn(SECRET, json.dumps(event))
        self.assertNotIn("headers", event)

    def test_dispatch_evidence_distinguishes_dns_unknown_and_http_response(self):
        failures = [
            (urllib.error.URLError(socket.gaierror(socket.EAI_NONAME, "fixture DNS")), False, False),
            (socket.gaierror(socket.EAI_NONAME, "fixture DNS"), False, False),
            (urllib.error.URLError(TimeoutError("fixture timeout")), False, True),
            (OSError("fixture disconnect"), False, True),
            (urllib.error.HTTPError("https://fixture.invalid", 503, "fixture", {}, io.BytesIO()), True, False),
        ]
        for error, received, unknown in failures:
            with self.subTest(error=type(error).__name__):
                client = self.client()
                before = client.usage.to_dict()
                endpoint_ = Endpoint(error)
                with self.wire(endpoint_), self.assertRaises(ProviderError):
                    client.chat(MESSAGES)
                event = self.events[-1]
                self.assertTrue(event["dispatch_attempted"])
                self.assertIs(event["response_received"], received)
                self.assertIs(event["request_sent"], received)
                self.assertIs(event["sent_unknown"], unknown)
                self.assertEqual(len(endpoint_.requests), 1)
                self.assertEqual(client.usage.requests, before["requests"] + 1)
                self.assertEqual(client.usage.errors, before["errors"] + 1)
        with self.wire(Endpoint(native())):
            self.client().chat(MESSAGES)
        event = self.events[-1]
        self.assertTrue(event["dispatch_attempted"])
        self.assertTrue(event["response_received"])
        self.assertTrue(event["request_sent"])
        self.assertFalse(event["sent_unknown"])

    def test_missing_credentials_records_body_without_reserving(self):
        client = self.client(api_key_env="PROVIDERS_MISSING_OFFLINE_KEY")
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(MissingCredential):
            client.chat(MESSAGES)
        self.assertEqual(client.usage.requests, 0)
        self.assertEqual(len(self.events), 1)
        self.assertIsNotNone(self.events[0]["request_body_sha256"])
        self.assertFalse(self.events[0]["request_sent"])
        self.assertFalse(self.events[0]["dispatch_attempted"])
        self.assertFalse(self.events[0]["response_received"])
        self.assertFalse(self.events[0]["sent_unknown"])

    def test_echoed_credential_redacted_in_full_content(self):
        endpoint_ = Endpoint(native([answer("visible " + SECRET)]))
        with self.wire(endpoint_):
            result = self.client().chat(MESSAGES + [{"role": "user", "content": SECRET}])
        self.assertIn(SECRET, result["choices"][0]["message"]["content"])
        self.assertNotIn(SECRET, json.dumps(self.events))
        event = self.events[-1]
        self.assertTrue(event["replay_limited"])
        self.assertTrue(event["record_changed"])
        paths = {r["path"] for r in event["redactions"]}
        self.assertIn("/request/input/3/content", paths)
        self.assertIn("/response/output/0/content/0/text", paths)
        for redaction in event["redactions"]:
            self.assertEqual(redaction["kind"], "credential")
            self.assertNotIn("sha256", redaction)
            self.assertGreater(redaction["len"], 0)

    def test_credential_field_redaction_metadata_avoids_secret_hashes(self):
        original = {"request": {"api_key": SECRET, "content": "prefix " + SECRET},
                    "response": {"reasoning_content": SECRET}, "events": []}
        recorded = redact_record(original, (SECRET,))
        self.assertEqual(original["request"]["api_key"], SECRET)
        self.assertTrue(recorded["replay_limited"])
        self.assertTrue(recorded["record_changed"])
        self.assertEqual(len(recorded["redactions"]), 3)
        self.assertNotIn(SECRET, json.dumps(recorded))
        self.assertNotIn(hashlib.sha256(SECRET.encode()).hexdigest(), json.dumps(recorded))
        credential = next(r for r in recorded["redactions"] if r["path"] == "/request/api_key")
        self.assertEqual(credential["len"], len(SECRET))
        clean = redact_record({"request": {"content": "safe"}})
        self.assertFalse(clean["replay_limited"])
        self.assertFalse(clean["record_changed"])
        self.assertEqual(clean["redactions"], [])

    def test_failed_terminal_tools_are_not_executable_but_raw_audit_survives(self):
        partial = {**function(arguments='{"query":'), "status": "incomplete"}
        refused = {"type": "message", "content": [{"type": "refusal", "refusal": "refused"}]}
        for raw, finish in [
            (native([partial], "incomplete", incomplete_details={"reason": "max_output_tokens"}), "length"),
            (native([partial], "incomplete", incomplete_details={"reason": "content_filter"}), "incomplete"),
            (native([function(), refused]), "content_filter"),
        ]:
            for stream in (False, True):
                with self.subTest(finish=finish, stream=stream):
                    client = self.client()
                    before = client.usage.to_dict()
                    wire = sse({"type": "response." + raw["status"], "response": raw}) if stream else raw
                    with self.wire(Endpoint(wire)):
                        result = (client.chat_stream(MESSAGES, on_event=lambda *a, **kw: None)
                                  if stream else client.chat(MESSAGES))
                    self.assertEqual(result["choices"][0]["finish_reason"], finish)
                    self.assertEqual(client.tool_calls(result), [])
                    self.assertNotIn("tool_calls", client.message(result))
                    self.assertEqual(self.events[-1]["response"], raw)
                    if stream:
                        self.assertEqual(self.events[-1]["events"][-1]["response"], raw)
                    self.assertEqual(client.usage.errors, before["errors"] + 1)
                    self.assertEqual(client.usage.input_tokens, before["input_tokens"] + 7)

    def test_completed_response_with_incomplete_native_call_fails_closed(self):
        raw = native([{**function(), "status": "incomplete"}])
        with self.wire(Endpoint(raw)), self.assertRaises(ProviderError):
            self.client().chat(MESSAGES)
        self.assertEqual(self.events[-1]["response"], raw)

    def test_incomplete_refusal_empty_failed_never_success(self):
        cases = [(native([answer("partial")], "incomplete",
                         incomplete_details={"reason": "max_output_tokens"}), "length"),
                 (native([], "incomplete", incomplete_details={"reason": "content_filter"}), "incomplete"),
                 (native([{"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}]),
                  "content_filter"),
                 (native([]), None), (native([], "failed", error={"message": SECRET}), None)]
        for payload, finish in cases:
            client = self.client()
            before = client.usage.errors
            endpoint_ = Endpoint(payload)
            with self.subTest(finish=finish), self.wire(endpoint_):
                if finish:
                    result = client.chat(MESSAGES)
                    self.assertEqual(result["choices"][0]["finish_reason"], finish)
                else:
                    with self.assertRaises(ProviderError):
                        client.chat(MESSAGES)
            self.assertEqual(client.usage.errors, before + 1)
            self.assertIsNotNone(self.events[-1]["error"])

    def test_responses_stream_crlf_multicall_and_hidden_reasoning(self):
        raw = native([{"type": "reasoning", "id": "rs", "encrypted_content": "opaque",
                      "summary": []}, answer("preface"), function(), function("call_b")])
        events = [
            {"type": "response.reasoning_text.delta", "delta": "hidden stream thought"},
            {"type": "response.output_text.delta", "delta": "preface"},
            {"type": "response.output_item.added", "output_index": 2, "item": function(arguments="")},
            {"type": "response.function_call_arguments.delta", "output_index": 2, "delta": '{"query":'},
            {"type": "response.function_call_arguments.done", "output_index": 2, "arguments": '{"query":"x"}'},
            {"type": "response.output_item.done", "output_index": 3, "item": function("call_b")},
            {"type": "response.completed", "response": raw}]
        endpoint_ = Endpoint(sse(*events))
        client, callbacks = self.client(), []
        with self.wire(endpoint_):
            result = client.chat_stream(MESSAGES, tools=TOOLS,
                        on_event=lambda name, **fields: callbacks.append((name, fields)))
        self.assertEqual(client.message(result)["_response_items"], raw["output"])
        self.assertEqual(len(client.tool_calls(result)), 2)
        self.assertEqual(callbacks, [("thinking", {"chars": len("hidden stream thought")}),
                                     ("content", {"text": "preface"})])
        self.assertTrue(endpoint_.bodies[0]["stream"])
        self.assertEqual(client.usage.output_tokens, 3)
        self.assertNotIn("hidden stream thought", json.dumps(self.events))
        self.assertEqual(len(self.events[0]["events"]), len(events))

    def test_sse_callbacks_happen_before_terminal_is_read(self):
        callbacks = []
        class InteractiveResponse:
            status = 200
            def __enter__(inner): return inner
            def __exit__(inner, *exc): return False
            def __iter__(inner):
                yield b'data: {"type":"response.output_text.delta","delta":"live"}\n'
                yield b'\n'
                self.assertEqual(callbacks, ["live"])
                yield b'data: ' + json.dumps({"type": "response.completed", "response": native()}).encode() + b'\n'
                yield b'\n'
        with self.wire(lambda request, **kw: InteractiveResponse()):
            self.client().chat_stream(MESSAGES, on_event=lambda name, **kw: callbacks.append(kw["text"]))

    def test_stream_cannot_drop_observed_tool_or_arguments(self):
        for observed, terminal in [
            ({"type": "response.output_item.done", "output_index": 1, "item": function()}, native()),
            ({"type": "response.function_call_arguments.done", "output_index": 0, "arguments": "{}"},
             native([function()])),
            ({"type": "response.refusal.delta", "delta": "refused"}, native()),
            ({"missing_type": True}, native()),
        ]:
            with self.subTest(observed=observed), self.wire(Endpoint(sse(observed,
                    {"type": "response.completed", "response": terminal}))), \
                    self.assertRaises(StreamProtocolError):
                self.client().chat_stream(MESSAGES, on_event=lambda *a, **kw: None)

    def test_stream_refusal_and_error_event_are_not_success(self):
        refused = native([{"type": "message", "content": [{"type": "refusal", "refusal": "refused"}]}])
        client, callbacks = self.client(), []
        with self.wire(Endpoint(sse({"type": "response.refusal.delta", "delta": "refused"},
                                   {"type": "response.completed", "response": refused}))):
            result = client.chat_stream(MESSAGES, on_event=lambda *a, **kw: callbacks.append((a, kw)))
        self.assertEqual(result["choices"][0]["finish_reason"], "content_filter")
        self.assertEqual(callbacks, [])
        with self.wire(Endpoint(sse({"type": "error", "message": "echo " + SECRET,
                                    "usage": {"input_tokens": 5}}))), self.assertRaises(StreamProtocolError):
            client.chat_stream(MESSAGES, on_event=lambda *a, **kw: None)
        self.assertEqual(client.usage.to_dict(), {"requests": 2, "input_tokens": 12,
                                                "output_tokens": 3, "errors": 2})
        self.assertNotIn(SECRET, json.dumps(self.events))

    def test_broken_stream_retains_usage_and_partial_events(self):
        class BrokenResponse:
            status = 200
            def __enter__(inner): return inner
            def __exit__(inner, *exc): return False
            def __iter__(inner):
                yield b'data: {"type":"response.output_text.delta","delta":"partial","usage":{"input_tokens":11}}\n'
                yield b'\n'
                raise OSError("disconnected " + SECRET)
        client = self.client()
        with self.wire(lambda request, **kw: BrokenResponse()), self.assertRaises(ProviderError):
            client.chat_stream(MESSAGES, on_event=lambda *a, **kw: None)
        self.assertEqual(client.usage.to_dict(), {"requests": 1, "input_tokens": 11,
                                                "output_tokens": 0, "errors": 1})
        self.assertIn("partial", json.dumps(self.events))
        self.assertNotIn(SECRET, json.dumps(self.events))

    def test_stream_without_terminal_or_malformed_retains_partial(self):
        first = {"type": "response.output_text.delta", "delta": "partial preserved"}
        for wire in (sse(first), sse(first, "{bad"), sse("[DONE]")):
            client = self.client()
            before = client.usage.errors
            with self.subTest(wire=wire), self.wire(Endpoint(wire)), self.assertRaises(StreamProtocolError):
                client.chat_stream(MESSAGES, on_event=lambda *a, **kw: None)
            self.assertEqual(client.usage.errors, before + 1)
            if b"partial" in wire:
                self.assertIn("partial preserved", json.dumps(self.events[-1]))

    def test_terminal_without_final_blank_line(self):
        event = {"type": "response.completed", "response": native()}
        with self.wire(Endpoint(("data: " + json.dumps(event)).encode())):
            result = self.client().chat_stream(MESSAGES, on_event=lambda *a, **kw: None)
        self.assertEqual(result["choices"][0]["finish_reason"], "stop")

    def test_stream_failed_and_incomplete_usage_counted_once(self):
        for status in ("failed", "incomplete"):
            client = self.client()
            before = client.usage.to_dict()
            raw = native([answer("partial")], status,
                         incomplete_details={"reason": "max_output_tokens"})
            with self.subTest(status=status), self.wire(Endpoint(sse({"type": "response." + status,
                                                                    "response": raw}))):
                if status == "failed":
                    with self.assertRaises(ProviderError):
                        client.chat_stream(MESSAGES, on_event=lambda *a, **kw: None)
                else:
                    result = client.chat_stream(MESSAGES, on_event=lambda *a, **kw: None)
                    self.assertEqual(result["choices"][0]["finish_reason"], "length")
            self.assertEqual(client.usage.input_tokens, before["input_tokens"] + 7)
            self.assertEqual(client.usage.errors, before["errors"] + 1)

    def test_cancellation_accounts_observed_usage_once(self):
        # Usage on an earlier frame survives a callback raising BaseException.
        raw = sse({"type": "response.in_progress", "response": {"usage": {"input_tokens": 9}}},
                  {"type": "response.output_text.delta", "delta": "partial"})
        client = self.client()
        def cancel(*args, **kwargs):
            raise KeyboardInterrupt()
        with self.wire(Endpoint(raw)), self.assertRaises(KeyboardInterrupt):
            client.chat_stream(MESSAGES, on_event=cancel)
        self.assertEqual(client.usage.to_dict(), {"requests": 1, "input_tokens": 9,
                                                "output_tokens": 0, "errors": 1})
        self.assertEqual(len(self.events), 1)

    def test_responses_stream_requires_terminal_not_done_marker(self):
        frames = [{"type": "response.reasoning_text.delta", "delta": "hidden Responses"},
                  {"type": "response.output_text.delta", "delta": "live"}]
        raw = native([answer("live")], usage={"input_tokens": 5, "output_tokens": 2})
        client = self.client()
        with self.wire(Endpoint(sse(*frames, {"type": "response.completed", "response": raw}))):
            result = client.chat_stream(MESSAGES, on_event=lambda *a, **kw: None)
        self.assertEqual(client.content(result), "live")
        self.assertNotIn("hidden Responses", json.dumps(self.events))
        for tail in ((), ("[DONE]",)):
            with self.wire(Endpoint(sse(*frames, *tail))), self.assertRaises(StreamProtocolError):
                client.chat_stream(MESSAGES, on_event=lambda *a, **kw: None)
        self.assertEqual(client.usage.to_dict(), {"requests": 3, "input_tokens": 5,
                                                "output_tokens": 2, "errors": 2})

    def test_explicit_effort_presets_are_sent_without_forced_high(self):
        for effort in ("none", "low", "medium", "high", "xhigh", "max"):
            for role in ("main", "retrieval", "summary"):
                with self.subTest(effort=effort, role=role):
                    client = create_client(self.cfg, self.profile(parameters={"reasoning": {"effort": effort}}),
                                           role=role)
                    endpoint_ = Endpoint(native())
                    with self.wire(endpoint_):
                        client.chat(MESSAGES)
                    self.assertEqual(endpoint_.bodies[0]["reasoning"], {"effort": effort})
                    self.assertTrue(endpoint_.requests[0].full_url.endswith("/responses"))

    def test_unsupported_effort_http_error_has_no_fallback(self):
        client = self.client(parameters={"reasoning": {"effort": "none"}})
        error = urllib.error.HTTPError("https://fixture.invalid/responses", 400,
                    "unsupported effort", {}, io.BytesIO(b'{"error":"unsupported effort"}'))
        endpoint_ = Endpoint(error)
        with self.wire(endpoint_), self.assertRaises(ProviderError):
            client.chat(MESSAGES)
        self.assertEqual(len(endpoint_.requests), 1)
        self.assertEqual(endpoint_.bodies[0]["reasoning"], {"effort": "none"})
        self.assertEqual(self.events[-1]["http_status"], 400)

    def test_codex_constructor_does_not_read_credentials_and_chat_aggregates_real_sse(self):
        headers = {"Authorization": "Bearer " + SECRET, "ChatGPT-Account-Id": "fixture-account",
                   "OpenAI-Beta": "responses=experimental"}
        with patch("continuity.codex_credentials.resolve_codex_headers", return_value=headers) as resolve:
            client = self.client(auth="codex_subscription", api_key_env=None,
                                 base_url="https://chatgpt.com/backend-api/codex")
            resolve.assert_not_called()
            endpoint_ = Endpoint(sse({"type": "response.completed", "response": native()}))
            with self.wire(endpoint_):
                result = client.chat(MESSAGES, tools=TOOLS)
            resolve.assert_called_once_with("https://chatgpt.com/backend-api/codex/responses")
        self.assertEqual(client.content(result), "visible")
        self.assertTrue(endpoint_.bodies[0]["stream"])
        self.assertEqual(endpoint_.bodies[0]["instructions"], "")
        self.assertEqual(endpoint_.bodies[0]["tools"][0]["name"], "lookup")
        self.assertNotIn("service_tier", endpoint_.bodies[0])
        self.assertNotIn(SECRET, json.dumps(self.events))
        self.assertNotIn("fixture-account", json.dumps(self.events))
        # Secret headers are intentionally not inherited on HTTP redirects.
        self.assertNotIn("Authorization", endpoint_.requests[0].headers)
        self.assertEqual(endpoint_.requests[0].get_header("Authorization"), "Bearer " + SECRET)

    def test_codex_and_api_profiles_preserve_identical_roles_context_tools_and_effort(self):
        headers = {"Authorization": "Bearer " + SECRET, "ChatGPT-Account-Id": "fixture-account"}
        parameters = {"temperature": None, "reasoning": {"effort": "none"}}
        api = self.client(parameters=parameters)
        subscription = self.client(parameters=parameters, auth="codex_subscription", api_key_env=None,
                                    base_url="https://chatgpt.com/backend-api/codex")
        endpoint_ = Endpoint(native(), sse({"type": "response.completed", "response": native()}))
        with self.wire(endpoint_), patch("continuity.codex_credentials.resolve_codex_headers", return_value=headers):
            for client in (api, subscription):
                client.chat(MESSAGES, tools=TOOLS)
        for request in endpoint_.requests:
            self.assertTrue(request.full_url.endswith("/responses"))
        for field in ("input", "tools", "tool_choice", "reasoning", "store", "include"):
            self.assertEqual(endpoint_.bodies[0][field], endpoint_.bodies[1][field])
        self.assertEqual(endpoint_.bodies[0]["input"], MESSAGES)
        self.assertEqual(endpoint_.bodies[1]["input"], MESSAGES)
        self.assertEqual(endpoint_.bodies[1]["reasoning"], {"effort": "none"})

    def test_codex_credential_error_maps_to_missing_credential(self):
        from continuity.codex_credentials import CodexCredentialError
        client = self.client(auth="codex_subscription", base_url="https://chatgpt.com/backend-api/codex")
        with patch("continuity.codex_credentials.resolve_codex_headers",
                   side_effect=CodexCredentialError("Codex credentials unavailable")), \
                self.assertRaises(MissingCredential):
            client.chat(MESSAGES)
        self.assertFalse(self.events[0]["request_sent"])
        self.assertEqual(client.usage.requests, 0)

    def test_codex_endpoint_and_auth_path_cannot_be_overridden(self):
        for base in ("https://fixture.invalid", "http://chatgpt.com/backend-api/codex",
                     "https://chatgpt.com/backend-api/codex/responses?key=x"):
            with self.subTest(base=base), self.assertRaises(ValueError):
                self.client(auth="codex_subscription", base_url=base)
        with self.assertRaises(ValueError):
            self.client(auth="codex_subscription", base_url="https://chatgpt.com/backend-api/codex",
                        codex_auth_path="/arbitrary/path")


if __name__ == "__main__":
    unittest.main()
