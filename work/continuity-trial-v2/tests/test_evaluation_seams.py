"""Offline checks that evaluations use the production chat loop safely."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from continuity import chat, db, memory_agent
from continuity.config import Config
from continuity.deepseek import Usage
from continuity.summary_store import Published
from continuity.provider_stream import StreamProtocolError
from continuity.providers import ProviderError, create_client


NOW = 1_800_000_000.0


def reply(content="", calls=None, items=None):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    if items is not None:
        message["_response_items"] = items
    return {"choices": [{"message": message,
                         "finish_reason": "tool_calls" if calls else "stop"}]}


def call(name, args, cid="call_1"):
    return {"id": cid, "type": "function", "function": {
        "name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


class ScriptClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []
        self.usage = Usage()

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": copy.deepcopy(messages), **kwargs})
        return next(self.replies)

    @staticmethod
    def message(payload):
        return payload["choices"][0]["message"]

    @staticmethod
    def content(payload):
        return ScriptClient.message(payload).get("content") or ""


class EvaluationSeamsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = Config(project_root=self.root, db_path=self.root / "chat.sqlite",
                             runs_dir=self.root / "runs", maint_state_dir=self.root / "memory",
                             usage_file=self.root / "usage.json")
        self.config.ensure_dirs()
        self.published = Published(True, version=7, text="原样记忆\n" + "细节" * 12_000)
        self.connections = []

    def tearDown(self):
        for conn in self.connections:
            conn.close()
        self.temp.cleanup()

    def connection(self, name):
        conn = db.connect(self.root / name)
        db.init_db(conn)
        conn.execute("INSERT INTO chat_log(created_at,instance,conversation_id,role,content) "
                     "VALUES(?,?,?,?,?)", (NOW - 10, "trial", "evaluation", "user", "前一条消息"))
        conn.commit()
        self.connections.append(conn)
        return conn

    def test_default_chat_and_injected_evaluation_have_identical_model_input(self):
        normal = ScriptClient([reply("正常回答")])
        evaluated = ScriptClient([reply("测试回答")])
        with patch.object(chat, "DeepSeekClient", return_value=normal), \
             patch.object(chat.time, "time", return_value=NOW):
            chat.chat_agentic(self.config, self.connection("normal.sqlite"), "同一问题",
                              published=self.published, conversation_id="evaluation")
        chat.chat_agentic(self.config, self.connection("evaluated.sqlite"), "同一问题",
                          published=self.published, conversation_id="evaluation",
                          client=evaluated, now_epoch=NOW)
        self.assertEqual(normal.calls, evaluated.calls)
        messages = evaluated.calls[0]["messages"]
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertIn(self.published.text, messages[0]["content"])
        self.assertIn("[user]", messages[0]["content"])
        self.assertIn("前一条消息", messages[0]["content"])
        for conn in self.connections:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM maint_queue").fetchone()[0], 0)

    def test_none_mode_rejects_unsolicited_tool_call_without_retrieving(self):
        client = ScriptClient([reply(calls=[call("delegate_memory", {"gaps": ["隐私"]})]),
                               reply("不知道你的历史。")])
        conn = self.connection("none.sqlite")
        with patch.object(chat.agents, "memory_agent_search",
                          side_effect=AssertionError("disabled tools executed")):
            result = chat.chat_agentic(self.config, conn, "你知道什么？",
                                       published=Published(False), client=client,
                                       conversation_id="evaluation", now_epoch=NOW,
                                       allow_memory_tools=False)
        self.assertEqual(result["answer"], "不知道你的历史。")
        self.assertTrue(all(c["tools"] is None for c in client.calls))
        self.assertNotIn(chat.AGENT_SYSTEM_ADDENDUM, client.calls[0]["messages"][0]["content"])
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM maint_queue").fetchone()[0], 0)

    def test_different_main_provider_uses_fixed_retrieval_client_and_real_history(self):
        conn = self.connection("pipeline.sqlite")
        conn.execute("INSERT INTO messages(conversation_id,node_id,role,text,create_time,"
                     "is_selected_branch,is_assistant_reasoning) VALUES(?,?,?,?,?,?,?)",
                     ("fixture", "past", "user", "我的合成测试暗号是青松。", NOW - 100, 1, 0))
        conn.commit()
        main_items = [{"type": "reasoning", "encrypted_content": "opaque-main"},
                      {"type": "function_call", "call_id": "call_1",
                       "name": "delegate_memory", "arguments": '{"gaps":["暗号"]}'}]
        main = ScriptClient([reply(calls=[call("delegate_memory", {"gaps": ["暗号"]})],
                                         items=main_items), reply("暗号是青松。")])
        retrieval_items = [{"type": "reasoning", "encrypted_content": "opaque-retrieval"},
                           {"type": "function_call", "call_id": "search_1",
                            "name": "search_history", "arguments": '{"query":"暗号"}'}]
        retrieval = ScriptClient([
            reply(calls=[call("search_history", {"query": "暗号"}, "search_1")],
                  items=retrieval_items),
            reply("已找到。"),
            reply(json.dumps({"status": "found", "findings": [{
                "claim": "暗号是青松", "kind": "user_fact",
                "source_ids": ["msg:fixture#past"]}],
                "evidence_source_ids": ["msg:fixture#past"]}, ensure_ascii=False))])
        result = chat.chat_agentic(self.config, conn, "我的暗号？", published=self.published,
                                   conversation_id="evaluation", now_epoch=NOW,
                                   client=main, retrieval_client=retrieval)
        self.assertEqual(result["answer"], "暗号是青松。")
        self.assertEqual(len(main.calls), 2)
        self.assertEqual(len(retrieval.calls), 3)
        self.assertEqual(retrieval.calls[1]["messages"][2]["_response_items"], retrieval_items)
        resumed = main.calls[1]["messages"]
        self.assertEqual(resumed[-2]["_response_items"], main_items)
        self.assertEqual(resumed[-1]["role"], "tool")
        handoff = json.loads(resumed[-1]["content"])
        self.assertEqual(handoff["evidence"][0]["text"], "我的合成测试暗号是青松。")
        self.assertIn("SIMULATED NOW", retrieval.calls[0]["messages"][1]["content"])
        self.assertNotIn("暗号是青松。", retrieval.calls[0]["messages"][1]["content"])

    def test_none_mode_does_not_request_tool_protocol_correction(self):
        client = ScriptClient([reply("<delegate_memory><queries>查找</queries></delegate_memory>"),
                               reply("我没有这段历史。")])
        chat.chat_agentic(self.config, self.connection("none-xml.sqlite"), "回忆一下",
                          published=Published(False), client=client, now_epoch=NOW,
                          conversation_id="evaluation", allow_memory_tools=False)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[-1]["messages"][-1]["content"], chat.RECOVERY_NUDGE)
        self.assertTrue(all(c["tools"] is None for c in client.calls))

    def test_retrieval_does_not_execute_incomplete_tool_or_accept_incomplete_synthesis(self):
        partial_tool = reply(calls=[call("search_history", {"query": "暗号"})])
        partial_tool["choices"][0]["finish_reason"] = "incomplete"
        partial_json = reply('{"status":"found","findings":[]}')
        partial_json["choices"][0]["finish_reason"] = "incomplete"
        conn = self.connection("incomplete-retrieval.sqlite")
        with patch.object(memory_agent, "_dispatch",
                          side_effect=AssertionError("incomplete tool dispatched")):
            for script in ([partial_tool], [reply("已查看"), partial_json]):
                with self.subTest(synthesis=len(script) == 2):
                    with self.assertRaises(StreamProtocolError):
                        memory_agent.run(self.config, conn, "trial", {"question": "暗号？"}, NOW,
                                         client=ScriptClient(script), max_tool_turns=1)

    def test_failed_http_output_is_recorded_with_diagnostic_and_credentials_redacted(self):
        secret = "synthetic-secret-not-a-real-key"
        body = json.dumps({"error": {"param": "temperature", "code": "unsupported_parameter",
                          "message": "unsupported parameter; echo " + secret}}).encode()
        error = urllib.error.HTTPError("https://fixture.invalid", 400, "fixture", {}, io.BytesIO(body))
        events = []
        client = create_client(self.config, {"id": "error", "model": "fixture",
            "base_url": "https://fixture.invalid", "api_format": "responses",
            "api_key_env": "EVALUATION_REVIEW_KEY", "parameters": {}}, recorder=events.append)
        with patch.dict("os.environ", {"EVALUATION_REVIEW_KEY": secret}), \
             patch("urllib.request.urlopen", side_effect=error), self.assertRaises(ProviderError):
            client.chat([{"role": "user", "content": "only a synthetic test"}])
        self.assertEqual(events[0]["response"]["error"]["param"], "temperature")
        self.assertEqual(events[0]["http_status"], 400)
        self.assertNotIn(secret, json.dumps(events))
        self.assertTrue(events[0]["record_changed"])
        self.assertEqual(client.usage.to_dict()["errors"], 1)


if __name__ == "__main__":
    unittest.main()
