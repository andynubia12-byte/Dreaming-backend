#!/usr/bin/env python3
"""Local synthetic transport fixture, never forwards requests to a provider."""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port=0):
        super().__init__(('127.0.0.1', port), Handler)
        self.requests = []
        self.lock = threading.Lock()

    @property
    def url(self):
        return f'http://127.0.0.1:{self.server_port}'


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        with self.server.lock:
            self.server.requests.append({'path': self.path, 'body': body})
        callback = getattr(self.server, 'before_response', None)
        if callback:
            callback(body)
        if not self.path.endswith('/responses') or 'messages' in body:
            self.send_error(400, 'Responses input required')
            return
        messages = body['input']
        tools = [m for m in messages if m.get('role') == 'tool' or m.get('type') == 'function_call_output']
        text = '\n'.join(str(m.get('content', '')) for m in messages)
        if 'FAIL_FIXTURE' in text:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b'fixture failure, synthetic-credential-must-never-persist')
            return
        message = {'role': 'assistant', 'content': 'Synthetic answer.'}

        def call(name, args):
            return {'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': f'fixture-call-{len(tools)}', 'type': 'function',
                 'function': {'name': name, 'arguments': json.dumps(args)}}]}

        if 'FORCE_TOOL_FIXTURE' in text and '请直接用简体中文' not in text:
            message = call('delegate_memory', {'gaps': ['must never execute with tools disabled']})
        elif body['model'].startswith('retrieval'):
            if body.get('text', {}).get('format', {}).get('type') == 'json_object':
                evidence = None
                for tool in tools:
                    result = json.loads(tool.get('content', tool.get('output', '{}')))
                    evidence = result.get('item') or evidence
                sid = evidence['source_id'] if evidence else None
                message['content'] = json.dumps({
                    'status': 'found' if sid else 'not_found',
                    'findings': ([{'claim': 'synthetic fixture evidence', 'kind': 'user_fact',
                                   'source_ids': [sid]}] if sid else []),
                    'evidence_source_ids': [sid] if sid else []})
            elif not tools:
                message = call('search_history', {'query': 'cobalt'})
            elif len(tools) == 1:
                results = json.loads(tools[0].get('content', tools[0].get('output', '{}'))).get('results', [])
                if results:
                    message = call('read_message', {'source_id': results[0]['source_id']})
        elif body.get('tools') and not tools:
            message = call('delegate_memory', {'gaps': ['cobalt fixture evidence']})
        elif tools:
            result = json.loads(tools[-1].get('content', tools[-1].get('output', '{}')))
            message['content'] = 'Synthetic evidence: ' + '\n'.join(e['text'] for e in result.get('evidence', []))

        output = []
        for tool in message.get('tool_calls', []):
            output.append({'type': 'function_call', 'call_id': tool['id'],
                           'name': tool['function']['name'], 'arguments': tool['function']['arguments']})
        if not output:
            output.append({'type': 'message', 'role': 'assistant', 'content': [
                {'type': 'output_text', 'text': message['content']}]})
        payload = {'id': 'synthetic-response', 'status': 'completed', 'output': output,
                   'usage': {'input_tokens': 7, 'output_tokens': 3}}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=18765)
    args = parser.parse_args()
    with FakeServer(args.port) as server:
        print(server.url, flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
