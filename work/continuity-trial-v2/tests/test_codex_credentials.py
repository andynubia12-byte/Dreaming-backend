"""Offline Codex credential checks using only temporary synthetic files."""

from __future__ import annotations

import base64
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
import unittest
import urllib.request
import uuid
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from continuity import codex_credentials as credentials


BASE_URL = "https://chatgpt.com/backend-api/codex"
NOW = 1_800_000_000


def jwt(claims):
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"synthetic.{payload}.signature"


class CodexCredentialsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.auth_path = self.root / "explicit-auth.json"
        self.package_root = self.root / "package"
        self.enterContext(patch("continuity.config.PROJECT_ROOT", self.package_root))
        self.home = self.root / "home"
        self.codex_home = self.root / "codex-home"
        self.enterContext(patch.dict(os.environ, {"CODEX_HOME": str(self.codex_home)}))
        self.enterContext(patch.object(Path, "home", return_value=self.home))
        self.enterContext(patch.object(credentials.time, "time", return_value=NOW))

    def write_auth(self, data=None, *, path=None):
        path = self.auth_path if path is None else path
        path.parent.mkdir(parents=True, exist_ok=True)
        if data is None:
            data = {"tokens": {"access_token": "synthetic-opaque-token",
                               "account_id": "synthetic-account",
                               "refresh_token": "synthetic-refresh-secret"}}
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def resolve(self):
        return credentials.resolve_codex_headers(BASE_URL, auth_path=self.auth_path)

    def assert_safe_error(self, action, *secrets):
        with self.assertRaises(credentials.CodexCredentialError) as caught:
            action()
        error = caught.exception
        rendered = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        self.assertIn("package-local", str(error))
        for secret in secrets:
            self.assertNotIn(secret, str(error))
            self.assertNotIn(secret, rendered)

    def test_import_does_not_resolve_or_read_credentials(self):
        with patch.object(Path, "read_text", side_effect=AssertionError("credential read")) as read, \
             patch.object(Path, "home", side_effect=AssertionError("home resolution")) as home:
            importlib.reload(credentials)
        read.assert_not_called()
        home.assert_not_called()

    def test_validation_is_credential_free_for_both_allowed_urls(self):
        with patch.object(Path, "read_text", side_effect=AssertionError("credential read")) as read, \
             patch.object(Path, "home", side_effect=AssertionError("home resolution")):
            credentials.validate_codex_base_url(BASE_URL)
            credentials.validate_codex_base_url(BASE_URL + "/responses")
        read.assert_not_called()

    def test_rejects_other_urls_before_any_credential_resolution(self):
        invalid = (
            "http://chatgpt.com/backend-api/codex",
            "https://example.com/backend-api/codex",
            "https://chatgpt.com.evil.example/backend-api/codex",
            "https://chatgpt.com@evil.example/backend-api/codex",
            "https://synthetic-userinfo-secret@chatgpt.com/backend-api/codex",
            "https://chatgpt.com:443/backend-api/codex",
            "https://chatgpt.com:/backend-api/codex",
            "https://chatgpt.com/backend-api/codex?key=synthetic-query-secret",
            "https://chatgpt.com/backend-api/codex?",
            "https://chatgpt.com/backend-api/codex#fragment",
            "https://chatgpt.com/backend-api/codex#",
            "https://chatgpt.com/backend-api/codex/",
            "https://chatgpt.com/backend-api/codex/responses/",
            "https://chatgpt.com/backend-api/codex/responses/other",
            "https://chatgpt.com/backend-api/codex/../responses",
            "https://chatgpt.com/backend-api/%63odex",
            "https://chatgpt.com/v1/responses",
            " https://chatgpt.com/backend-api/codex",
            "https://chatgpt.com/backend-api/codex\n",
            "",
        )
        with patch.object(Path, "read_text", side_effect=AssertionError("credential read")) as read, \
             patch.object(Path, "home", side_effect=AssertionError("home resolution")), \
             patch.object(credentials.os.environ, "get", side_effect=AssertionError("env resolution")):
            for url in invalid:
                with self.subTest(url=url), self.assertRaises(ValueError) as caught:
                    credentials.resolve_codex_headers(url)
                self.assertNotIn("synthetic-", str(caught.exception))
        read.assert_not_called()

    def test_explicit_path_headers_match_source_without_refresh_token(self):
        self.write_auth()
        headers = self.resolve()
        session = headers.pop("session_id")
        self.assertEqual(str(uuid.UUID(session)), session)
        self.assertEqual(headers, {
            "Authorization": "Bearer synthetic-opaque-token",
            "ChatGPT-Account-Id": "synthetic-account",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "OpenAI-Beta": "responses=experimental",
            "originator": "codex_cli_rs",
            "User-Agent": "codex_cli_rs/0.160.1",
        })
        self.assertNotIn("synthetic-refresh-secret", json.dumps(headers))

    def test_exact_responses_endpoint_is_supported(self):
        self.write_auth()
        headers = credentials.resolve_codex_headers(BASE_URL + "/responses", auth_path=self.auth_path)
        self.assertEqual(headers["Authorization"], "Bearer synthetic-opaque-token")

    def test_optional_account_id_is_omitted(self):
        for value in (None, ""):
            with self.subTest(value=value):
                self.write_auth({"tokens": {"access_token": "opaque", "account_id": value}})
                self.assertNotIn("ChatGPT-Account-Id", self.resolve())
        self.write_auth({"tokens": {"access_token": "opaque"}})
        self.assertNotIn("ChatGPT-Account-Id", self.resolve())

    def test_old_codex_home_is_ignored(self):
        self.write_auth(path=self.codex_home / "auth.json")
        with self.assertRaises(credentials.CodexCredentialError):
            credentials.resolve_codex_headers(BASE_URL)

    def test_default_path_is_package_local(self):
        self.write_auth(path=self.package_root / "secrets" / "codex-auth.json")
        with patch.object(Path, "home", side_effect=AssertionError("home discovery")):
            self.assertEqual(credentials.resolve_codex_headers(BASE_URL)["Authorization"],
                             "Bearer synthetic-opaque-token")

    def test_explicit_path_takes_precedence_over_environment(self):
        self.write_auth()
        self.write_auth({"tokens": {"access_token": "other-token"}},
                        path=self.codex_home / "auth.json")
        self.assertEqual(self.resolve()["Authorization"], "Bearer synthetic-opaque-token")

    def test_missing_file_error_is_safe(self):
        self.assert_safe_error(self.resolve, str(self.auth_path))

    def test_permission_error_is_safe_and_suppresses_exception_context(self):
        with patch.object(Path, "read_text", side_effect=PermissionError("synthetic-permission-secret")):
            self.assert_safe_error(self.resolve, "synthetic-permission-secret")

    def test_malformed_json_and_encoding_errors_are_safe(self):
        self.auth_path.write_text("synthetic-file-content-secret", encoding="utf-8")
        self.assert_safe_error(self.resolve, "synthetic-file-content-secret")
        self.auth_path.write_bytes(b"\xffsynthetic-file-content-secret")
        self.assert_safe_error(self.resolve, "synthetic-file-content-secret")

    def test_invalid_auth_shapes_and_missing_tokens_are_safe(self):
        for value in (None, [], "synthetic-shape-secret", 1, {},
                      {"tokens": []}, {"tokens": "synthetic-shape-secret"},
                      {"tokens": {}}, {"tokens": {"access_token": None}},
                      {"tokens": {"access_token": ""}}, {"tokens": {"access_token": 42}}):
            with self.subTest(value=value):
                self.auth_path.write_text(json.dumps(value), encoding="utf-8")
                self.assert_safe_error(self.resolve, "synthetic-shape-secret")

    def test_invalid_header_values_fail_safely(self):
        for field in ("access_token", "account_id"):
            for value in ("synthetic-header-secret\r\nInjected: yes", "contains space", "\x00", "\x7f", "中文"):
                with self.subTest(field=field, value=value):
                    tokens = {"access_token": "opaque", "account_id": "account"}
                    tokens[field] = value
                    self.write_auth({"tokens": tokens})
                    self.assert_safe_error(self.resolve, "synthetic-header-secret")

    def test_expired_jwt_fails_without_refresh_or_file_changes(self):
        for expiry in (NOW - 1, NOW, 0, str(NOW - 1)):
            with self.subTest(expiry=expiry):
                token = jwt({"exp": expiry})
                self.write_auth({"tokens": {"access_token": token,
                                            "refresh_token": "synthetic-refresh-secret"}})
                before = self.auth_path.read_bytes()
                self.assert_safe_error(self.resolve, token, "synthetic-refresh-secret")
                self.assertEqual(self.auth_path.read_bytes(), before)

    def test_near_expiry_is_not_refreshed_or_rejected(self):
        token = jwt({"exp": NOW + 1})
        self.write_auth({"tokens": {"access_token": token}})
        self.assertEqual(self.resolve()["Authorization"], f"Bearer {token}")

    def test_future_jwt_and_unknown_or_opaque_tokens_are_allowed(self):
        for token in (jwt({"exp": NOW + 3600}), jwt({}), jwt({"exp": "unknown"}),
                      jwt({"exp": None}), jwt({"exp": False}), jwt([]), jwt("claims"),
                      "synthetic-opaque-token", "synthetic.%%%.signature", "synthetic._w.signature"):
            with self.subTest(token=token):
                self.write_auth({"tokens": {"access_token": token}})
                self.assertEqual(self.resolve()["Authorization"], f"Bearer {token}")

    def test_repeated_calls_read_current_file_and_create_fresh_sessions(self):
        self.write_auth()
        before = self.auth_path.read_bytes()
        mtime = self.auth_path.stat().st_mtime_ns
        first, second = self.resolve(), self.resolve()
        self.assertNotEqual(first["session_id"], second["session_id"])
        self.assertEqual(self.auth_path.read_bytes(), before)
        self.assertEqual(self.auth_path.stat().st_mtime_ns, mtime)
        self.write_auth({"tokens": {"access_token": "replacement-token"}})
        self.assertEqual(self.resolve()["Authorization"], "Bearer replacement-token")

    def test_header_casing_is_compatible_with_urllib_request(self):
        self.write_auth()
        request = urllib.request.Request(BASE_URL + "/responses", headers=self.resolve(), method="POST")
        headers = {key.lower(): value for key, value in request.header_items()}
        self.assertEqual(headers["authorization"], "Bearer synthetic-opaque-token")
        self.assertEqual(headers["chatgpt-account-id"], "synthetic-account")
        self.assertEqual(headers["accept"], "text/event-stream")
        self.assertEqual(headers["openai-beta"], "responses=experimental")


if __name__ == "__main__":
    unittest.main()
