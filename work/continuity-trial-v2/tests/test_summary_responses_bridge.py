"""Real localhost Responses HTTP across the frozen engine's three stages."""
from __future__ import annotations

from dataclasses import replace
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

import test_sandbox_summary as fixtures
from test_sandbox_summary import ENGINE, hashes
from continuity.events import EventHub, Observer, observe
from continuity.providers import response_input
from continuity.sandbox_summary import SandboxSummary
from continuity.summary_store import Published
from memmaint.prompts import load_prompts


LOCAL_CLI = '''
import os, sys
sys.path.insert(0, ALIAS_PARENT)
from run_update_with_citation_alias import install, main
# Subscription tests use synthetic headers and a localhost endpoint. Production
# endpoint validation and credential resolution remain unchanged.
if os.environ.get('SUMMARY_FIXTURE_SUBSCRIPTION'):
    import continuity.codex_credentials as credentials
    credentials.validate_codex_base_url = lambda url: None
    credentials.resolve_codex_headers = lambda url: {'Authorization': 'Bearer fixture-subscription'}
'''


def stage_response(payload, *, repair=False):
    if 'new_messages' in payload:
        msg = next(m for m in payload['new_messages'] if m['role'] == 'user')
        quote = msg['text']
        operation = {'op': 'add', 'kind': 'preference', 'text': quote,
                     'evidence_message_ids': [msg['id']],
                     'quotes': [{'message_id': msg['id'], 'quote': quote}]}
        if repair and 'previous_attempt' not in payload:
            operation['quotes'][0]['quote'] = 'invented nonverbatim evidence'
        content = json.dumps({'operations': [operation]})
    elif 'claims' in payload:
        content = json.dumps({'verdicts': [dict(index=c['index'], verdict='supported',
                            role_ok=True, time_ok=True, overgeneralized=False,
                            issue='', corrected_text=None) for c in payload['claims']]})
    else:
        content = '# Local synthetic summary\n\n' + payload['memory_material']
        for alias, canonical in sorted(payload['reference_aliases'].items(), key=lambda p: -len(p[1])):
            content = content.replace('[' + canonical + ']', '[' + alias + ']')
    return {'id': 'resp_local', 'status': 'completed', 'output': [
        {'type': 'reasoning', 'content': [{'type': 'reasoning_text', 'text': 'PRIVATE-COT fixture-key'}]},
        {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': content}]}],
        'usage': {'input_tokens': 17, 'output_tokens': 11}}


class SummaryResponsesBridgeTests(unittest.TestCase):
    connection = fixtures.SandboxSummaryTests.connection
    turn = fixtures.SandboxSummaryTests.turn
    source = fixtures.SandboxSummaryTests.source

    def setUp(self):
        fixtures.SandboxSummaryTests.setUp(self)
        self.cli.write_text(LOCAL_CLI.replace('ALIAS_PARENT', repr(str(ENGINE / 'review-v2'))))
        env = patch.dict(os.environ, {'SUMMARY_LOCAL_HTTP_KEY': 'fixture-key'})
        env.start()
        self.addCleanup(env.stop)
        self.requests, self.raw_requests, self.paths = [], [], []
        self.failure = None
        self.repair = False
        self.gate = None
        self.gate_observed = False
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                raw = self.rfile.read(int(self.headers['Content-Length']))
                body = json.loads(raw)
                owner.paths.append(self.path)
                owner.requests.append(body)
                owner.raw_requests.append(raw)
                if self.path != '/responses':
                    self.send_error(404)
                    return
                payload = json.loads(body['input'][1]['content'])
                if owner.gate is not None and 'claims' in payload:
                    owner.gate_observed = owner.gate.wait(2)
                response = stage_response(payload, repair=owner.repair)
                if owner.failure == 'http':
                    self.send_response(400)
                    response = {'error': {'message': 'fixture-key provider refusal',
                                          'reasoning_content': 'PRIVATE-COT'}}
                else:
                    self.send_response(200)
                    if owner.failure == 'incomplete':
                        response.update(status='incomplete', incomplete_details={'reason': 'max_output_tokens'})
                if body['stream']:
                    event = {'type': 'response.' + response['status'], 'response': response}
                    raw_response = ('data: ' + json.dumps(event) + '\n\n').encode()
                    self.send_header('Content-Type', 'text/event-stream')
                else:
                    raw_response = json.dumps(response).encode()
                    self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(raw_response)))
                self.end_headers()
                self.wfile.write(raw_response)

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.url = f'http://127.0.0.1:{server.server_port}'
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def close_server():
            server.shutdown()
            server.server_close()
            thread.join()

        self.addCleanup(close_server)

    def config(self, name='session'):
        return replace(fixtures.SandboxSummaryTests.config(self, name), api_base_url=self.url,
                       api_key_env='SUMMARY_LOCAL_HTTP_KEY')

    def profile(self, effort='none'):
        return {'id': 'pool-model', 'model': 'arbitrary-pool-model', 'base_url': self.url,
                'api_format': 'responses', 'api_key_env': 'SUMMARY_LOCAL_HTTP_KEY',
                'parameters': {'reasoning': {'effort': effort}, 'max_output_tokens': 4096}}

    def summary(self, config=None, **kwargs):
        return SandboxSummary(config or self.config(), options={
            'enabled': True, 'max_requests': 9, **kwargs})

    def test_three_stages_exact_prompts_context_live_wire_and_single_ledger(self):
        config = self.config()
        conn = self.connection(config)
        summary = self.summary(config)
        events, hub = [], EventHub()
        self.gate = threading.Event()

        def recorder(event):
            events.append(event)
            self.gate.set()

        with observe(Observer(hub, session_id='responses')):
            result = summary.update(conn, *self.turn(conn), recorder=recorder)
        self.assertEqual(result['status'], 'ok', result)
        self.assertTrue(self.gate_observed, 'propose wire must arrive before verify completes')
        self.assertEqual(self.paths, ['/responses'] * 3)
        self.assertEqual(result['usage']['delta'], dict(requests=3, input_tokens=51, output_tokens=33, errors=0))
        self.assertFalse(config.usage_file.exists(), 'no second chat-account ledger')
        self.assertEqual(len(events), 3)
        self.assertIn('maintenance_publish_result', [e['phase'] for e, _ in hub.events])
        prompts = load_prompts()
        batch = json.loads(Path(result['batch_path']).read_text())
        for index, (body, event) in enumerate(zip(self.requests, events)):
            self.assertEqual(event['request'], body)
            self.assertEqual(event['request_body_sha256'], hashlib.sha256(self.raw_requests[index]).hexdigest())
            self.assertEqual(event['api_format'], 'responses')
            self.assertEqual(event['role'], 'summary')
            self.assertTrue(event['request_sent'])
            self.assertEqual([m['role'] for m in body['input']], ['system', 'user'])
            self.assertEqual(body['input'], response_input(body['input']))
            self.assertEqual(body['reasoning'], {'effort': 'high'})
            self.assertEqual(body['max_output_tokens'], 64000)
            self.assertFalse(body['store'])
            for forbidden in ('messages', 'previous_response_id', 'max_tokens', 'response_format'):
                self.assertNotIn(forbidden, body)
        for index, name in enumerate(('propose', 'verify')):
            self.assertEqual(self.requests[index]['input'][0]['content'], prompts[name])
            self.assertEqual(self.requests[index]['text'], {'format': {'type': 'json_object'}})
        self.assertNotIn('text', self.requests[-1])
        summary_system = self.requests[-1]['input'][0]['content']
        self.assertTrue(summary_system.startswith(prompts['summarize'] + '\n\n'))
        self.assertIn('reference_aliases', json.loads(self.requests[-1]['input'][1]['content']))
        proposed = json.loads(self.requests[0]['input'][1]['content'])
        self.assertEqual(proposed['new_messages'], batch['messages'])
        wire = Path(result['wire']).read_text()
        for secret in ('PRIVATE-COT', 'fixture-key'):
            self.assertNotIn(secret, wire)
        self.assertTrue(all(e['record_changed'] and e['replay_limited'] for e in events))

    def test_selected_effort_presets_and_omitted_default_are_never_forced_high(self):
        for effort in ('none', 'low', 'medium', 'high', 'xhigh', 'max', None):
            with self.subTest(effort=effort):
                config = self.config('effort-' + str(effort))
                profile = self.profile(effort)
                if effort is None:
                    profile['parameters'].pop('reasoning')
                summary = self.summary(config, profile=profile, model='legacy-value-ignored')
                conn = self.connection(config)
                before = len(self.requests)
                result = summary.update(conn, *self.turn(conn))
                self.assertEqual(result['status'], 'ok', result)
                self.assertEqual(summary.status()['profile'], profile)
                for body in self.requests[before:]:
                    self.assertEqual(body['model'], 'arbitrary-pool-model')
                    self.assertEqual(body['max_output_tokens'], 4096)
                    self.assertEqual(body.get('reasoning'), None if effort is None else {'effort': effort})
                reloaded = SandboxSummary(config, options={'enabled': True, 'max_requests': 9})
                self.assertEqual(reloaded.initialize()['profile'], profile)
                self.assertEqual(reloaded.model, profile['model'])
                self.assertEqual(profile['parameters'].get('reasoning'),
                                 None if effort is None else {'effort': effort})

    def test_subscription_profile_uses_shared_sse_transport_with_explicit_none(self):
        profile = self.profile()
        profile.update(auth='codex_subscription', model='gpt-6.1-sol')
        profile.pop('api_key_env')
        profile['parameters']['temperature'] = None
        config = self.config()
        conn = self.connection(config)
        events = []
        with patch('continuity.codex_credentials.validate_codex_base_url'), \
             patch.dict(os.environ, {'SUMMARY_FIXTURE_SUBSCRIPTION': '1'}):
            summary = self.summary(config, profile=profile)
            result = summary.update(conn, *self.turn(conn), recorder=events.append)
        self.assertEqual(result['status'], 'ok', result)
        self.assertEqual(self.paths, ['/responses'] * 3)
        self.assertEqual(result['usage']['model_calls'], 3)
        self.assertEqual(len(events), 3)
        for body, event in zip(self.requests, events):
            self.assertTrue(body['stream'])
            self.assertEqual(body['model'], 'gpt-6.1-sol')
            self.assertEqual(body['reasoning'], {'effort': 'none'})
            self.assertNotIn('temperature', body)
            self.assertEqual(event['auth'], 'codex_subscription')
            self.assertEqual(event['events'][0]['type'], 'response.completed')
        self.assertNotIn('fixture-subscription', Path(result['wire']).read_text())

    def test_http_and_incomplete_failures_keep_answer_no_retry_or_protocol_fallback(self):
        for failure in ('http', 'incomplete'):
            with self.subTest(failure=failure):
                config = self.config(failure)
                conn = self.connection(config)
                initial = Published(True, 0, 'Retained fixed summary.')
                summary = SandboxSummary(config, initial=initial, options={
                    'enabled': True, 'max_requests': 9, 'profile': self.profile()})
                self.failure = failure
                start = len(self.requests)
                result = summary.update(conn, *self.turn(conn))
                self.assertEqual(result['status'], 'failed', result)
                self.assertEqual(self.paths[start:], ['/responses'])
                self.assertEqual(result['usage']['model_calls'], 1)
                self.assertEqual(result['usage']['delta']['errors'], 1)
                self.assertEqual(summary.published(), initial)
                self.assertEqual(result['before_text'], result['after_text'])
                self.assertEqual(conn.execute('SELECT content FROM chat_log ORDER BY id DESC').fetchone()[0],
                                 'Synthetic answer retained.')
                self.assertEqual(self.requests[-1]['reasoning'], {'effort': 'none'})
                for folder in (config.maint_state_dir, Path(result['work_dir'])):
                    for path in folder.rglob('*'):
                        if path.suffix in ('.json', '.jsonl'):
                            self.assertNotIn('fixture-key', path.read_text(), str(path))
                            self.assertNotIn('PRIVATE-COT', path.read_text(), str(path))

    def test_semantic_repair_remains_bounded_and_all_attempts_accounted_once(self):
        self.repair = True
        config = self.config()
        conn = self.connection(config)
        summary = self.summary(config, profile=self.profile())
        result = summary.update(conn, *self.turn(conn))
        self.assertEqual(result['status'], 'ok', result)
        self.assertEqual(self.paths, ['/responses'] * 4)
        self.assertEqual(result['usage']['delta'], dict(requests=4, input_tokens=68, output_tokens=44, errors=0))
        self.assertEqual(len(Path(result['wire']).read_text().splitlines()), 4)
        payloads = [json.loads(b['input'][1]['content']) for b in self.requests]
        self.assertIn('previous_attempt', payloads[1])
        self.assertEqual(payloads[0]['new_messages'], payloads[1]['new_messages'])

    def test_runtime_metadata_migration_keeps_seed_ledger_and_historical_artifacts(self):
        source, initial = self.source()
        config = self.config()
        conn = self.connection(config)
        summary = SandboxSummary(config, source_config=source, initial=initial,
                                 options={'enabled': True, 'max_requests': 9})
        summary.initialize()
        path = config.maint_state_dir / 'sandbox-summary.json'
        original = json.loads(path.read_text())
        original.pop('profile')
        original.pop('model')
        path.write_text(json.dumps(original))
        ledger_before = (config.maint_state_dir / 'ledger/usage.json').read_bytes()
        source_before = hashes(source.maint_state_dir)
        historical = config.maint_state_dir / 'ledger/historical.jsonl'
        historical.write_text('{"error":"old artifact stays byte identical", "reasoning_content":"old marker"}\n')
        old_bytes = historical.read_bytes()
        reloaded = SandboxSummary(config, source_config=source, initial=initial,
                                  options={'enabled': True, 'max_requests': 9})
        status = reloaded.initialize()
        self.assertEqual(status['capability'], 'responses')
        self.assertEqual((config.maint_state_dir / 'ledger/usage.json').read_bytes(), ledger_before)
        migrated = json.loads(path.read_text())
        self.assertEqual({k: migrated[k] for k in original}, original)
        self.assertEqual(migrated['profile']['api_format'], 'responses')
        self.turn(conn, 'Source chat rows already represented by the cloned seed.')
        result = reloaded.update(conn, *self.turn(conn))
        self.assertEqual(result['status'], 'ok', result)
        self.assertEqual(result['usage']['baseline']['requests'], 298)
        self.assertEqual(result['usage']['total']['requests'], 301)
        self.assertEqual(historical.read_bytes(), old_bytes)
        self.assertEqual(hashes(source.maint_state_dir), source_before)
        call_log = config.maint_state_dir / 'ledger/calls.jsonl'
        call_prefix = call_log.read_bytes()
        immutable = {p: p.read_bytes() for p in config.maint_state_dir.glob('users/*/versions/**/*')
                     if p.is_file()}
        # Existing saved CC metadata migrates explicitly; new CC input is rejected.
        legacy = self.profile()
        legacy.update(api_format='chat_completions')
        legacy['parameters'] = {'max_tokens': 2048, 'thinking': {'type': 'disabled'}}
        with self.assertRaisesRegex(ValueError, 'Responses'):
            self.summary(config, profile=legacy)
        metadata = json.loads(path.read_text())
        metadata['profile'] = legacy
        path.write_text(json.dumps(metadata))
        resumed = SandboxSummary(config, source_config=source, initial=initial,
                                 options={'enabled': True, 'max_requests': 9})
        self.assertEqual(resumed.initialize()['profile']['parameters'],
                         {'max_output_tokens': 2048, 'reasoning': {'effort': 'none'}})
        next_result = resumed.update(conn, *self.turn(conn, '我喜欢另一个合成测试颜色。'))
        self.assertEqual(next_result['status'], 'ok', next_result)
        self.assertEqual(next_result['usage']['total']['requests'], 304)
        self.assertEqual(self.requests[-1]['reasoning'], {'effort': 'none'})
        self.assertEqual(historical.read_bytes(), old_bytes)
        self.assertTrue(call_log.read_bytes().startswith(call_prefix))
        for path, original_bytes in immutable.items():
            self.assertEqual(path.read_bytes(), original_bytes, str(path))


if __name__ == '__main__':
    unittest.main()
