"""Responses-only loopback fixture; synthetic mechanics, never a quality judge."""
from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time


@contextmanager
def fixture_server(*, stream_delay=0.0, expected_keys=None):
    requests = []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            expected = (expected_keys or {}).get(body.get("model"))
            with lock:
                requests.append({"path": self.path, "body": body,
                    "authorization_valid": (self.headers.get("Authorization") == "Bearer " + expected)
                    if expected is not None else None})
            # A wrong runtime route must fail, even if it carries a Responses body.
            if self.path not in ("/responses", "/v1/responses"):
                self.send_error(400, "Only the Responses API is accepted")
                return
            if "messages" in body or not isinstance(body.get("input"), list):
                self.send_error(400, "Responses input items are required")
                return
            if (body.get("store") is not False or body.get("include") != ["reasoning.encrypted_content"]
                    or "service_tier" in body):
                self.send_error(400, "Canonical stateless Responses options are required")
                return
            history = body["input"]
            history_text = json.dumps(history, ensure_ascii=False)
            tools = body.get("tools") or []
            if any(t.get("type") != "function" or "function" in t or not t.get("name") for t in tools):
                self.send_error(400, "Responses function tools must be flat")
                return
            names = [t["name"] for t in tools]
            returned = any(m.get("type") == "function_call_output" for m in history)
            tool_name = None
            if "delegate_memory" in names and not returned:
                tool_name, arguments = "delegate_memory", {"gaps": ["机械测试暗号"]}
            elif "search_history" in names and not returned:
                tool_name, arguments = "search_history", {"query": "机械测试暗号"}
            if tool_name:
                output = [{"type": "function_call", "id": "fc_fixture",
                           "call_id": "fixture_call", "name": tool_name,
                           "arguments": json.dumps(arguments, ensure_ascii=False), "status": "completed"}]
            else:
                if "retrieval" in body["model"] and not tools:
                    content = json.dumps({"status": "found", "findings": [{
                        "claim": "机械测试暗号是青松", "kind": "user_fact",
                        "source_ids": ["msg:fixture#past"]}],
                        "evidence_source_ids": ["msg:fixture#past"]}, ensure_ascii=False)
                elif "retrieval" in body["model"]:
                    content = "离线夹具：已查到，等待综合。"
                else:
                    content = "离线夹具结果：" + ("青松" if "青松" in history_text else "没有记忆")
                output = [{"type": "message", "id": "msg_fixture", "role": "assistant",
                           "status": "completed", "content": [
                               {"type": "output_text", "text": content, "annotations": []}]}]
            payload = {"id": "resp_fixture", "object": "response", "status": "completed",
                       "model": body["model"], "output": output,
                       "usage": {"input_tokens": 12, "output_tokens": 5, "total_tokens": 17}}
            if body.get("stream"):
                events = [{"type": "response.created", "response": {
                    "id": payload["id"], "status": "in_progress"}}]
                for index, item in enumerate(output):
                    events.append({"type": "response.output_item.added", "output_index": index,
                                   "item": {**item, "status": "in_progress",
                                            **({"arguments": ""} if tool_name else {"content": []})}})
                    if item["type"] == "message":
                        text = item["content"][0]["text"]
                        events.extend({"type": "response.output_text.delta", "output_index": index,
                                       "content_index": 0, "delta": text[start:start + 3]}
                                      for start in range(0, len(text), 3))
                    else:
                        arguments_json = item["arguments"]
                        events.extend({"type": "response.function_call_arguments.delta", "output_index": index,
                                       "item_id": item["id"], "delta": arguments_json[start:start + 3]}
                                      for start in range(0, len(arguments_json), 3))
                        events.append({"type": "response.function_call_arguments.done", "output_index": index,
                                       "item_id": item["id"], "arguments": arguments_json})
                    events.append({"type": "response.output_item.done", "output_index": index, "item": item})
                events.append({"type": "response.completed", "response": payload})
                frames = [("event: " + e["type"] + "\r\ndata: " + json.dumps(
                    {**e, "sequence_number": i}, ensure_ascii=False) + "\r\n\r\n").encode()
                          for i, e in enumerate(events)]
                data = b"".join(frames)  # Native terminal event; never a CC [DONE] sentinel.
                content_type = "text/event-stream"
            else:
                data, content_type = json.dumps(payload, ensure_ascii=False).encode(), "application/json"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if body.get("stream"):
                for frame in frames:
                    self.wfile.write(frame)
                    self.wfile.flush()
                    if stream_delay:
                        time.sleep(stream_delay)
            else:
                self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
