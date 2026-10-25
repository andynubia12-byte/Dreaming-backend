"""Real loopback HTTP + synthetic providers; never reads real auth or calls APIs."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import http.client
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import uuid
from contextlib import closing
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))
from continuity import db, evaluation, workbench
from continuity.config import Config
from continuity.summary_store import Published
from evaluation_fixture_server import fixture_server

KEY = 'workbench-synthetic-key-never-persist'


class FakeSummary:
    updates = 0
    gate = None
    entered = None
    fail = False

    def __init__(self, config, *, source_config, initial, options):
        self.config, self.initial, self.options = config, initial, options
        self.path = config.maint_state_dir / 'fake-summary.json'

    def initialize(self):
        if not self.path.exists():
            workbench.write(self.path, dataclasses.asdict(self.initial))
        return {'seed_mode': 'synthetic_new_turn_branch'}

    def published(self):
        return Published(**workbench.read(self.path))

    def update(self, conn, uid, aid, *, recorder=None):
        type(self).updates += 1
        if self.entered:
            self.entered.set()
        if self.gate:
            self.gate.wait(5)
        if self.fail:
            raise RuntimeError(KEY)
        before = self.published()
        after = Published(True, version=before.version + 1, text=before.text + ' GENERATED_ONLY')
        workbench.write(self.path, dataclasses.asdict(after))
        return {'status': 'published', 'version': after.version, 'before_text': before.text,
                'after_text': after.text, 'usage': {'requests': 1}}


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        formal = self.root / 'formal'
        formal.mkdir()
        self.source = Config(project_root=formal, db_path=formal / 'chat.sqlite',
            runs_dir=formal / 'runs', usage_file=formal / 'usage.json',
            maint_state_dir=formal / 'state', user_id='synthetic-user', instance='synthetic-source',
            api_key_env='WORKBENCH_TEST_KEY')
        self.source.ensure_dirs()
        with closing(db.connect(self.source.db_path)) as conn:
            db.init_db(conn)
            conn.execute("INSERT INTO messages(conversation_id,node_id,message_id,role,text,create_time,"
                "is_selected_branch,is_assistant_reasoning) VALUES('fixture','past','past','user',?, ?,1,0)",
                ('机械测试暗号是青松', time.time() - 200))
            conn.execute("INSERT INTO chat_log(created_at,instance,conversation_id,role,content) VALUES(?,?,?,?,?)",
                (time.time() - 100, self.source.instance, 'formal-session', 'user', 'FORMAL_RECENT_SECRET'))
            conn.execute("INSERT INTO maint_queue(batch_id,status) VALUES('formal-pending','pending')")
            conn.commit()
            conn.execute('PRAGMA journal_mode=DELETE')
        summary = self.source.maint_state_dir / 'users/user/versions/1/summary.md'
        summary.parent.mkdir(parents=True)
        summary.write_text('FULL_SELECTED 青松', 'utf-8')
        workbench.write(summary.parents[2] / 'manifest.json', {'user_id': self.source.user_id,
            'version': 1, 'summary_path': str(summary),
            'summary_sha': hashlib.sha256(summary.read_bytes()).hexdigest()})
        self.baseline = self.formal_hashes()
        self.env = patch.dict(os.environ, {'WORKBENCH_TEST_KEY': KEY})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.fixture = fixture_server()
        self.provider_url, self.requests = self.fixture.__enter__()
        self.addCleanup(self.fixture.__exit__, None, None, None)
        self.managed = self.root / 'evaluations/workbench'
        self.server = workbench.serve_workbench(self.source, port=0, root=self.managed)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop)
        FakeSummary.updates, FakeSummary.gate, FakeSummary.entered, FakeSummary.fail = 0, None, None, False

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def formal_hashes(self):
        return {str(p.relative_to(self.source.project_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.source.project_root.rglob('*') if p.is_file()}

    def request(self, method, path, body=None, headers=None, raw=False):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=10)
        hdr = {'Content-Type': 'application/json', **(headers or {})}
        connection.request(method, path, body=json.dumps(body) if body is not None else None, headers=hdr)
        response = connection.getresponse()
        content = response.read().decode()
        status = response.status
        connection.close()
        return status, content if raw else json.loads(content)

    def profile(self, name='main', api_format='responses'):
        return {'id': name, 'model': 'retrieval-fixture' if name == 'retrieval' else 'synthetic-main',
            'base_url': self.provider_url, 'api_format': api_format,
            'api_key_env': 'WORKBENCH_TEST_KEY', 'parameters': {}, 'timeout_seconds': 3}

    def session(self, memory=None, retrieval=False, summary=False, api_format='responses'):
        status, body = self.request('POST', '/workbench/sessions', {
            'model': self.profile(api_format=api_format), 'memory': memory or {'mode': 'none'},
            'retrieval': {'enabled': retrieval, 'profile': self.profile('retrieval')},
            'summary': {'enabled': summary}})
        self.assertEqual(status, 201, body)
        return body['session']['id']

    def start(self, sid, message='问题', request_id=None):
        status, body = self.request('POST', f'/workbench/sessions/{sid}/chat',
            {'request_id': request_id or str(uuid.uuid4()), 'message': message})
        self.assertEqual(status, 202, body)
        return body['run_id']

    def finish(self, rid):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            status, body = self.request('GET', '/workbench/runs/' + rid)
            if body['run']['status'] not in ('running', 'accepted'):
                # run_complete is appended just after terminal registry persistence.
                status, sse = self.request('GET', f'/workbench/runs/{rid}/events', raw=True)
                return body['run'], [json.loads(line[6:]) for line in sse.splitlines() if line.startswith('data: ')]
            time.sleep(.01)
        self.fail('worker did not finish')

    def suite(self, contexts=None):
        return {'models': [self.profile()], 'questions': [{'id': 'q', 'text': '<script>青松</script>'}],
            'contexts': contexts or [{'id': 'none', 'mode': 'none'}],
            'budgets': {'max_requests': 20, 'trial_max_requests': 20, 'max_input_tokens': 1000000}}

    def batch_finish(self, bid):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            status, detail = self.request('GET', '/workbench/batches/' + bid)
            if detail['batch']['status'] != 'running':
                return detail
            time.sleep(.01)
        self.fail('batch did not finish')

    def test_bootstrap_zero_calls_no_snapshot_copy_or_auth_reads(self):
        with patch('continuity.codex_credentials.resolve_codex_headers', side_effect=AssertionError('auth read')):
            status, bootstrap = self.request('GET', '/workbench/bootstrap')
        self.assertEqual(status, 200)
        self.assertEqual(self.requests, [])
        self.assertEqual(list((self.managed / 'snapshots').iterdir()), [])
        self.assertEqual(bootstrap['defaults']['model_id'], 'deepseek-pro')
        self.assertIsNone(next(s for s in bootstrap['credential_status'] if s['id'] == 'codex-sol-high')['available'])
        self.assertEqual(self.formal_hashes(), self.baseline)
        self.assertNotIn(KEY, json.dumps(bootstrap))
        for asset in ('/', '/workbench.js', '/workbench.css'):
            if (ROOT / 'src/continuity' / ('workbench.html' if asset == '/' else asset[1:])).exists():
                self.assertEqual(self.request('GET', asset, raw=True)[0], 200)

    def test_model_pool_shared_metadata_and_zero_bootstrap_auth_or_api(self):
        with patch('continuity.codex_credentials.resolve_codex_headers', side_effect=AssertionError('auth read')):
            status, pool = self.request('GET', '/workbench/models')
            self.assertEqual(status, 200, pool)
            status, boot = self.request('GET', '/workbench/bootstrap')
            self.assertEqual(status, 200, boot)
            self.assertEqual(pool['profiles'], boot['profiles'])
            self.assertEqual(pool['credential_status'], boot['credential_status'])
            self.assertIn(boot['defaults']['retrieval_profile']['id'], [p['id'] for p in pool['profiles']])
            custom = {**self.profile('pool-environment'), 'model': 'any-provider-model'}
            status, saved = self.request('POST', '/workbench/models', {'profile': custom, 'label': '共享配置'})
            self.assertEqual(status, 200, saved)
            self.assertEqual(saved['profile']['label'], '共享配置')
            pool = self.request('GET', '/workbench/models')[1]
            boot = self.request('GET', '/workbench/bootstrap')[1]
            self.assertEqual(pool['profiles'], boot['profiles'])
            self.assertEqual(pool['credential_status'], boot['credential_status'])
        allowed = {'id', 'model', 'base_url', 'api_format', 'api_key_env', 'auth', 'parameters',
                   'timeout_seconds', 'token_limit_field', 'label', 'builtin'}
        self.assertTrue(all(not (set(p) - allowed) for p in pool['profiles']))
        self.assertTrue(all(type(p['builtin']) is bool for p in pool['profiles']))
        self.assertFalse(next(p for p in pool['profiles'] if p['id'] == custom['id'])['builtin'])
        self.assertTrue(all(p['builtin'] for p in pool['profiles'] if p['id'] != custom['id']))
        self.assertNotIn(KEY, json.dumps(pool))
        self.assertEqual(self.requests, [])
        self.assertEqual(self.formal_hashes(), self.baseline)

    def test_responses_defaults_and_shared_summary_pool_profile_contract(self):
        boot = self.request('GET', '/workbench/bootstrap')[1]
        self.assertEqual(boot['capabilities']['api_format'], 'responses')
        for p in boot['profiles']:
            self.assertEqual(p['api_format'], 'responses')
            self.assertEqual(p['parameters']['reasoning']['effort'], 'high')
        self.assertEqual(workbench.summary_options(None)['profile']['api_format'], 'responses')
        self.assertEqual(boot['defaults']['summary_profile']['profile']['id'], 'deepseek-flash')
        custom = self.profile('shared-none')
        custom['parameters'] = {'reasoning': {'effort': 'none'}, 'max_output_tokens': 64000}
        status, saved = self.request('POST', '/workbench/models', {'profile': custom, 'label': 'All roles'})
        self.assertEqual(status, 200, saved)
        p = saved['profile']
        with patch('continuity.sandbox_summary.SandboxSummary', FakeSummary):
            status, result = self.request('POST', '/workbench/sessions', {
                'model': p, 'memory': {'mode': 'none'},
                'retrieval': {'enabled': False, 'profile': p},
                'summary': {'enabled': True, 'model': p['id']}})
            self.assertEqual(status, 201, result)
            session = result['session']
            clean = workbench.profile(p)
            self.assertEqual(session['model'], clean)
            self.assertEqual(session['retrieval']['profile'], clean)
            self.assertEqual(session['summary']['profile'], clean)
            self.assertEqual(session['summary']['model'], clean['model'])
            self.assertNotIn('label', session['summary']['profile'])
            self.finish(self.start(session['id']))
            self.assertEqual(self.requests[0]['body']['reasoning']['effort'], 'none')
            self.assertTrue(all(r['path'].endswith('/responses') for r in self.requests))
        status, body = self.request('POST', '/workbench/sessions', {
            'model': p, 'memory': {'mode': 'none'},
            'summary': {'enabled': False, 'model': 'mismatch', 'profile': p}})
        self.assertEqual(status, 400, body)

    def test_new_cc_is_rejected_on_all_inputs(self):
        legacy = {**self.profile(), 'api_format': 'chat_completions'}
        bodies = [('/workbench/models', {'profile': legacy}),
            ('/workbench/sessions', {'model': legacy, 'memory': {'mode': 'none'}}),
            ('/workbench/sessions', {'model': self.profile(), 'memory': {'mode': 'none'},
                'retrieval': {'enabled': False, 'profile': legacy}}),
            ('/workbench/sessions', {'model': self.profile(), 'memory': {'mode': 'none'},
                'summary': {'enabled': False, 'profile': legacy}}),
            ('/workbench/batches/plan', {'suite': {**self.suite(), 'models': [legacy]}})]
        for path, body in bodies:
            with self.subTest(path=path):
                status, result = self.request('POST', path, body)
                self.assertEqual(status, 400, result)
        self.assertEqual(self.requests, [])

    def test_legacy_session_load_continue_run_preserves_artifacts(self):
        sid = self.session()
        stored = copy.deepcopy(self.server.workbench.sessions[sid])
        for p in (stored['model'], stored['retrieval']['profile']):
            p['api_format'] = 'chat_completions'
            p['parameters'] = {'max_tokens': 1000, 'reasoning_effort': 'none'}
        stored['summary'].pop('profile')
        file = self.managed / 'sessions' / sid / 'session.json'
        workbench.write(file, stored)
        workspace = file.with_name('workspace.json')
        record = workbench.read(workspace)
        record['summary'].pop('profile')
        workbench.write(workspace, record)
        wire = self.managed / 'runs' / 'legacy-fixture' / 'wire.jsonl'
        wire.parent.mkdir()
        wire.write_text('{"api_format":"chat_completions","historical":true}\n')
        before = {f: f.read_bytes() for f in (file, workspace, wire)}
        restarted = workbench.Workbench(self.source, self.managed)
        loaded = restarted.detail(sid)['session']
        self.assertEqual(loaded['model']['api_format'], 'responses')
        self.assertEqual(loaded['model']['parameters']['reasoning']['effort'], 'none')
        self.assertEqual({m['role'] for m in loaded['protocol_migrations']}, {'main', 'retrieval', 'summary'})
        continued = restarted.continue_session({'continue_from': sid})['session']
        self.assertEqual(continued['model']['api_format'], 'responses')
        started = restarted.start_chat(sid, {'request_id': str(uuid.uuid4()), 'message': 'legacy continue'})
        deadline = time.monotonic() + 8
        while restarted.runs[started['run_id']]['status'] == 'running' and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertEqual(restarted.runs[started['run_id']]['status'], 'completed')
        self.assertEqual(self.requests[-1]['body']['reasoning']['effort'], 'none')
        self.assertTrue(all(r['path'].endswith('/responses') for r in self.requests))
        self.assertEqual({f: f.read_bytes() for f in before}, before)

    def test_legacy_batch_runtime_profiles_preserve_historical_results(self):
        bid = 'legacy-batch'
        folder = self.managed / 'batches' / bid
        folder.mkdir()
        stored = {'id': bid, 'status': 'completed', 'summary': {'enabled': False, 'model': 'deepseek-flash'}}
        workbench.write(folder / 'batch.json', stored)
        legacy = {**self.profile(), 'api_format': 'chat_completions',
                  'parameters': {'max_tokens': 1234, 'reasoning_effort': 'none'}}
        workbench.write(folder / 'run/inputs/suite.json', {'models': [legacy], 'retrieval_profile': legacy})
        workbench.write(folder / 'run/manifest.json', {'status': 'complete', 'historical': True})
        workbench.append(folder / 'run/results.jsonl', {'index': 0, 'parameters': legacy, 'answer': 'old answer'})
        before = {p: p.read_bytes() for p in folder.rglob('*') if p.is_file()}
        restarted = workbench.Workbench(self.source, self.managed)
        detail = restarted.batch_detail(bid)
        runtime = detail['batch']['runtime_profiles']
        self.assertEqual(runtime['main'][0]['parameters']['reasoning']['effort'], 'none')
        self.assertEqual(runtime['retrieval']['api_format'], 'responses')
        self.assertEqual(runtime['summary']['api_format'], 'responses')
        self.assertEqual(detail['results'][0]['parameters']['api_format'], 'chat_completions')
        self.assertEqual({m['role'] for m in detail['batch']['protocol_migrations']}, {'main', 'retrieval', 'summary'})
        self.assertEqual({p: p.read_bytes() for p in before}, before)
        self.assertEqual(self.requests, [])

    def test_summary_preflight_checks_selected_pool_credential_without_api_calls(self):
        selected = self.profile('summary-missing')
        selected['api_key_env'] = 'WORKBENCH_MISSING_SUMMARY_TEST_KEY'
        status, plan = self.request('POST', '/workbench/batches/plan', {
            'suite': self.suite(), 'summary': {'enabled': True, 'profile': selected}})
        self.assertEqual(status, 200, plan)
        self.assertFalse(plan['valid'])
        self.assertEqual(plan['missing_credentials'][-1]['role'], 'summary')
        sol = self.request('GET', '/workbench/models')[1]['profiles'][-1]
        status, plan = self.request('POST', '/workbench/batches/plan', {
            'suite': self.suite(), 'summary': {'enabled': True, 'profile': sol}})
        self.assertEqual(status, 200, plan)
        self.assertTrue(plan['valid'])
        self.assertIn(sol['id'], plan['runtime_auth_unchecked'])
        self.assertEqual(self.requests, [])

    def test_model_pool_temporary_key_real_responses_chat_batch_and_restart(self):
        temporary_key = 'pool-only-secret-' + uuid.uuid4().hex
        custom = {**self.profile('pool-responses', 'responses'), 'model': 'arbitrary-provider-model'}
        custom.pop('api_key_env')
        status, saved = self.request('POST', '/workbench/models', {
            'profile': custom, 'label': 'Responses 共享模型', 'api_key': temporary_key})
        self.assertEqual(status, 200, saved)
        selected = saved['profile']
        reference = selected['api_key_env']
        self.assertTrue(reference.startswith('WORKBENCH_POOL_KEY_'))
        self.assertEqual(os.getenv(reference), temporary_key)
        self.assertEqual(os.getenv('WORKBENCH_TEST_KEY'), KEY)
        self.assertNotIn(temporary_key, json.dumps(saved))
        retrieval_key = 'pool-retrieval-secret-' + uuid.uuid4().hex
        retrieval_profile = {**self.profile('pool-retrieval', 'responses'), 'model': 'retrieval-other-provider'}
        status, retrieval_saved = self.request('POST', '/workbench/models', {
            'profile': retrieval_profile, 'label': '独立 Responses 检索', 'api_key': retrieval_key})
        self.assertEqual(status, 200, retrieval_saved)
        retrieval = retrieval_saved['profile']
        retrieval_reference = retrieval['api_key_env']
        self.assertNotEqual(retrieval_reference, reference)
        self.assertEqual(os.getenv(retrieval_reference), retrieval_key)
        self.assertEqual(os.getenv(reference), temporary_key)
        status, response = self.request('POST', '/workbench/sessions', {
            'model': selected, 'memory': {'mode': 'none'}, 'retrieval': {'enabled': False},
            'summary': {'enabled': False}})
        self.assertEqual(status, 201, response)
        run, events = self.finish(self.start(response['session']['id']))
        self.assertEqual(run['status'], 'completed', events)
        wire = self.request('GET', f'/workbench/runs/{run["id"]}/wire')[1]['wire_events']
        self.assertTrue(any(e['api_format'] == 'responses' and e['request']['model'] == custom['model'] for e in wire))
        status, continued = self.request('POST', '/workbench/sessions', {
            'continue_from': response['session']['id'], 'retrieval': {'enabled': True, 'profile': retrieval}})
        self.assertEqual(status, 201, continued)
        run, events = self.finish(self.start(continued['session']['id']))
        self.assertEqual(run['status'], 'completed', events)
        self.assertIn('retrieval_handoff', [e['phase'] for e in events])
        wire = self.request('GET', f'/workbench/runs/{run["id"]}/wire')[1]['wire_events']
        self.assertEqual({e['role'] for e in wire}, {'main', 'retrieval'})
        self.assertTrue(all(e['api_format'] == 'responses' for e in wire))
        self.assertTrue(any(e['request']['model'] == retrieval['model'] for e in wire))
        suite = self.suite([{'id': 'retrieval-only', 'mode': 'pipeline', 'memory_id': 'current', 'inject_memory': False}])
        suite['models'] = [selected]
        suite['retrieval_profile'] = retrieval
        status, batch = self.request('POST', '/workbench/batches', {'suite': suite})
        self.assertEqual(status, 202, batch)
        detail = self.batch_finish(batch['batch']['id'])
        self.assertEqual(detail['batch']['status'], 'completed', detail)
        self.assertEqual(detail['results'][0]['status'], 'passed', detail)
        self.assertFalse(detail['results'][0]['context_injection_enabled'])
        self.assertIn('青松', detail['results'][0]['answer'])
        self.assertTrue(all(r['path'].endswith('/responses') for r in self.requests))
        self.assertEqual(self.request('GET', '/workbench/models')[1]['profiles'],
                         self.request('GET', '/workbench/bootstrap')[1]['profiles'])
        self.assertTrue((self.managed / 'models.json').is_file())
        for path in self.managed.rglob('*'):
            if path.is_file():
                self.assertNotIn(temporary_key.encode(), path.read_bytes(), str(path))
                self.assertNotIn(retrieval_key.encode(), path.read_bytes(), str(path))
        calls = len(self.requests)
        with patch.dict(os.environ):
            os.environ.pop(reference)
            os.environ.pop(retrieval_reference)
            restored = workbench.Workbench(self.source, self.managed)
            restored_pool = restored.models()
            self.assertEqual(next(p for p in restored_pool['profiles'] if p['id'] == custom['id']), selected)
            self.assertFalse(next(s for s in restored_pool['credential_status'] if s['id'] == custom['id'])['available'])
            self.assertFalse(next(s for s in restored_pool['credential_status'] if s['id'] == retrieval['id'])['available'])
            self.assertEqual(len(self.requests), calls)
        self.assertEqual(self.formal_hashes(), self.baseline)

    def test_model_pool_updates_delete_keep_session_snapshot_and_builtin_protection(self):
        original = {**self.profile('pool-edit'), 'model': 'original-model'}
        status, saved = self.request('POST', '/workbench/models', {'profile': original, 'label': '原模型',
            'api_key': 'old-session-key-' + uuid.uuid4().hex})
        self.assertEqual(status, 200, saved)
        status, response = self.request('POST', '/workbench/sessions', {
            'model': saved['profile'], 'memory': {'mode': 'none'}})
        self.assertEqual(status, 201, response)
        sid = response['session']['id']
        old_snapshot = self.request('GET', f'/workbench/sessions/{sid}')[1]['session']['model']
        updated = {**original, 'model': 'updated-model'}
        self.assertEqual(self.request('POST', '/workbench/models', {'profile': updated, 'label': '新模型'})[0], 200)
        self.assertEqual(self.request('GET', f'/workbench/sessions/{sid}')[1]['session']['model'], old_snapshot)
        status, deleted = self.request('DELETE', '/workbench/models/' + original['id'])
        self.assertEqual(status, 200, deleted)
        self.assertNotIn(original['id'], [p['id'] for p in self.request('GET', '/workbench/models')[1]['profiles']])
        self.assertEqual(self.request('GET', f'/workbench/sessions/{sid}')[1]['session']['model'], old_snapshot)
        run, events = self.finish(self.start(sid))
        self.assertEqual(run['status'], 'completed', events)
        self.assertEqual(self.requests[0]['body']['model'], original['model'])
        builtin = next(p for p in self.request('GET', '/workbench/models')[1]['profiles'] if p['id'] == 'deepseek-pro')
        edited_builtin = {**builtin, 'id': 'copied-pro', 'parameters': {'max_output_tokens': 12000}}
        self.assertEqual(self.request('POST', '/workbench/models', {'profile': edited_builtin})[0], 200)
        self.assertIn(self.request('DELETE', '/workbench/models/deepseek-pro')[0], (400, 403, 409))
        self.assertIn('deepseek-pro', [p['id'] for p in self.request('GET', '/workbench/models')[1]['profiles']])
        self.assertEqual(self.formal_hashes(), self.baseline)

    def test_model_pool_rejects_cross_origin_unknown_fields_and_secret_echo(self):
        private_key = 'must-not-echo-' + uuid.uuid4().hex
        initial = self.request('GET', '/workbench/models')[1]
        body = {'profile': self.profile('pool-reject'), 'api_key': private_key}
        for method, path in [('POST', '/workbench/models'), ('DELETE', '/workbench/models/deepseek-pro')]:
            status, result = self.request(method, path, body if method == 'POST' else None,
                                          headers={'Origin': 'https://evil.example'})
            self.assertEqual(status, 403, result)
            self.assertNotIn(private_key, json.dumps(result))
        invalid = copy.deepcopy(body)
        invalid['profile']['base_url'] = 'https://user:' + private_key + '@example.invalid'
        status, error = self.request('POST', '/workbench/models', invalid)
        self.assertEqual(status, 400, error)
        self.assertNotIn(private_key, json.dumps(error))
        status, error = self.request('POST', '/workbench/models', {**body, 'file': '/tmp/not-allowed'})
        self.assertEqual(status, 400, error)
        self.assertNotIn(private_key, json.dumps(error))
        for duplicate in ({**body, 'label': private_key},
                          {**body, 'profile': {**body['profile'], 'parameters': {'stop': private_key}}}):
            status, error = self.request('POST', '/workbench/models', duplicate)
            self.assertEqual(status, 400, error)
            self.assertNotIn(private_key, json.dumps(error))
        self.assertEqual(self.request('GET', '/workbench/models')[1], initial)
        self.assertEqual(self.requests, [])
        for path in self.managed.rglob('*'):
            if path.is_file():
                self.assertNotIn(private_key.encode(), path.read_bytes(), str(path))

    def test_full_literal_injection_and_recent_turns_no_formal_recent(self):
        full = '全文青松\n' + '长材料' * 30000 + '\nEND_FULL_CONTEXT'
        sid = self.session({'mode': 'context', 'text': full})
        run, events = self.finish(self.start(sid, '第一轮'))
        self.assertEqual(run['status'], 'completed')
        self.assertIn(full, self.requests[0]['body']['input'][0]['content'])
        self.assertNotIn('FORMAL_RECENT_SECRET', json.dumps(self.requests))
        self.assertNotIn('tools', self.requests[0]['body'])
        self.finish(self.start(sid, '第二轮'))
        self.assertIn('第一轮', self.requests[-1]['body']['input'][0]['content'])
        self.assertEqual(self.formal_hashes(), self.baseline)
        self.assertIn('answer_delta', [e['phase'] for e in events])
        self.assertEqual(events[-1]['phase'], 'run_complete')
        for path in self.managed.rglob('config.json'):
            config = workbench.restore_config(workbench.read(path))
            for field in dataclasses.fields(config):
                value = getattr(config, field.name)
                if isinstance(value, Path):
                    self.assertTrue(value.is_relative_to(self.managed), (field.name, value))

    def test_retrieval_independent_memory_off_responses_real_handoffs(self):
        main = self.profile()
        main['parameters'] = {'reasoning': {'effort': 'high'}}
        retrieval = self.profile('retrieval')
        retrieval['parameters'] = {'reasoning': {'effort': 'none'}}
        status, result = self.request('POST', '/workbench/sessions', {
            'model': main, 'memory': {'mode': 'none'},
            'retrieval': {'enabled': True, 'profile': retrieval}})
        self.assertEqual(status, 201, result)
        sid = result['session']['id']
        run, events = self.finish(self.start(sid))
        self.assertEqual(run['status'], 'completed', events)
        phases = [e['phase'] for e in events]
        self.assertIn('tool_call', phases)
        self.assertIn('retrieval_handoff', phases)
        wire = self.request('GET', f'/workbench/runs/{run["id"]}/wire')[1]['wire_events']
        self.assertTrue(any(e['role'] == 'retrieval' for e in wire))
        self.assertTrue(any('青松' in str(e['request']) for e in wire if e['role'] == 'main'))
        for event in wire:
            self.assertEqual(event['api_format'], 'responses')
            self.assertEqual(event['request']['reasoning']['effort'],
                             'none' if event['role'] == 'retrieval' else 'high')
        first = next(e for e in wire if e['role'] == 'main')['request']
        self.assertNotIn('FULL_SELECTED', str(first))
        self.assertNotIn(KEY, json.dumps(wire))
        config = workbench.restore_config(self.server.workbench.sessions[sid]['config'])
        with closing(db.connect(config.db_path)) as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM maint_queue').fetchone()[0], 0)
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM messages').fetchone()[0], 1)
        self.assertEqual(self.formal_hashes(), self.baseline)

    def test_disabled_injection_keeps_selected_summary_source(self):
        sid = self.session({'mode': 'context', 'enabled': False, 'id': 'current'}, retrieval=True)
        self.finish(self.start(sid))
        self.assertNotIn('FULL_SELECTED', json.dumps(self.requests))
        self.assertEqual(self.request('GET', f'/workbench/sessions/{sid}/memory')[1]['text'], 'FULL_SELECTED 青松')

    def test_summary_independent_injection_off_never_injects_new_memory(self):
        with patch('continuity.sandbox_summary.SandboxSummary', FakeSummary):
            sid = self.session({'mode': 'none', 'enabled': False}, summary=True)
            self.finish(self.start(sid, 'first'))
            self.finish(self.start(sid, 'second'))
            self.assertEqual(FakeSummary.updates, 2)
            self.assertNotIn('GENERATED_ONLY', json.dumps(self.requests))
            self.assertIn('GENERATED_ONLY', self.request('GET', f'/workbench/sessions/{sid}/memory')[1]['text'])
            self.assertFalse(any(r['body'].get('tools') for r in self.requests))

    def test_summary_on_injected_next_turn_and_lock_covers_maintenance(self):
        FakeSummary.gate, FakeSummary.entered = threading.Event(), threading.Event()
        with patch('continuity.sandbox_summary.SandboxSummary', FakeSummary):
            sid = self.session({'mode': 'context', 'text': 'baseline 青松'}, summary=True)
            rid = self.start(sid)
            self.assertTrue(FakeSummary.entered.wait(3))
            status, result = self.request('POST', f'/workbench/sessions/{sid}/chat',
                {'request_id': str(uuid.uuid4()), 'message': 'busy'})
            self.assertEqual(status, 409)
            detail = self.request('GET', f'/workbench/sessions/{sid}')[1]
            self.assertEqual(len(detail['messages']), 2)
            FakeSummary.gate.set()
            run, events = self.finish(rid)
            phases = [e['phase'] for e in events]
            self.assertLess(phases.index('answer_complete'), phases.index('maintenance_published'))
            self.assertLess(phases.index('maintenance_published'), phases.index('run_complete'))
            self.finish(self.start(sid, 'next'))
            self.assertIn('GENERATED_ONLY', self.requests[-1]['body']['input'][0]['content'])

    def test_summary_failure_preserves_answer_and_disabled_does_not_call(self):
        with patch('continuity.sandbox_summary.SandboxSummary', FakeSummary):
            sid = self.session(summary=False)
            self.finish(self.start(sid))
            self.assertEqual(FakeSummary.updates, 0)
            FakeSummary.fail = True
            sid = self.session({'mode': 'context', 'text': 'baseline'}, summary=True)
            run, events = self.finish(self.start(sid))
            self.assertEqual(run['status'], 'completed')
            self.assertIn('maintenance_failed', [e['phase'] for e in events])
            detail = self.request('GET', f'/workbench/sessions/{sid}')[1]
            self.assertEqual(detail['messages'][-1]['role'], 'assistant')
            self.assertEqual(detail['memory']['text'], 'baseline')

    def test_idempotency_conflicts_durable_reconnect_and_restart(self):
        sid, request_id = self.session(), str(uuid.uuid4())
        rid = self.start(sid, request_id=request_id)
        run, events = self.finish(rid)
        self.assertEqual(self.start(sid, request_id=request_id), rid)
        count = len(self.requests)
        conflict = self.request('POST', f'/workbench/sessions/{sid}/chat',
            {'request_id': request_id, 'message': 'different'})
        self.assertEqual(conflict[0], 409)
        cursor = events[-2]['seq']
        sse = self.request('GET', f'/workbench/runs/{rid}/events?after={cursor}', raw=True)[1]
        replay = [json.loads(line[6:]) for line in sse.splitlines() if line.startswith('data: ')]
        self.assertEqual([e['seq'] for e in replay], [events[-1]['seq']])
        reset_sse = self.request('GET', f'/workbench/runs/{rid}/events?after=999999', raw=True)[1]
        reset = next(json.loads(line[6:]) for line in reset_sse.splitlines() if line.startswith('data: '))
        self.assertEqual(reset['phase'], 'replay_gap')
        self.assertEqual(reset['session_id'], sid)
        self.assertEqual(reset['request_id'], request_id)
        restored = workbench.Workbench(self.source, self.managed)
        self.assertTrue(restored.start_chat(sid, {'request_id': request_id, 'message': '问题'})['reused'])
        self.assertEqual(len(restored.detail(sid)['messages']), 2)
        self.assertEqual(len(self.requests), count)

    def test_recovery_marks_interrupted_and_does_not_reexecute(self):
        sid = self.session()
        rid = uuid.uuid4().hex
        folder = self.managed / 'runs' / rid
        folder.mkdir()
        record = {'id': rid, 'session_id': sid, 'request_id': str(uuid.uuid4()),
            'message_sha256': hashlib.sha256(b'question').hexdigest(), 'status': 'running'}
        workbench.write(folder / 'run.json', record)
        (folder / 'events.jsonl').write_bytes(b'{torn')
        recovered = workbench.Workbench(self.source, self.managed)
        self.assertEqual(recovered.runs[rid]['status'], 'interrupted')
        self.assertEqual(workbench.jsonlines(folder / 'events.jsonl')[-1]['phase'], 'run_complete')
        self.assertEqual(self.requests, [])

    def test_provider_failure_never_saves_assistant_or_queue(self):
        sid = self.session()
        with patch('continuity.workbench.create_client', side_effect=RuntimeError(KEY)):
            run, events = self.finish(self.start(sid))
        self.assertEqual(run['status'], 'failed')
        self.assertNotIn(KEY, json.dumps(events))
        messages = self.request('GET', f'/workbench/sessions/{sid}')[1]['messages']
        self.assertFalse(any(m['role'] == 'assistant' for m in messages))

    def test_batch_uses_existing_module_freezes_once_and_previous_known_id(self):
        suite = self.suite([{'id': 'a', 'mode': 'snapshot', 'memory_id': 'current'},
                            {'id': 'b', 'mode': 'pipeline', 'memory_id': 'current'},
                            {'id': 'c', 'mode': 'context', 'text': '青松'}])
        suite['retrieval_profile'] = self.profile('retrieval')
        with patch.object(evaluation, 'create_snapshot', wraps=evaluation.create_snapshot) as freeze:
            status, planned = self.request('POST', '/workbench/batches/plan', {'suite': suite})
            self.assertEqual(status, 200, planned)
            self.assertEqual(freeze.call_count, 1)
            self.assertEqual(planned['api_calls'], 0)
            self.assertEqual(self.requests, [])
        with patch.object(evaluation, 'run_suite', wraps=evaluation.run_suite) as run_suite:
            status, response = self.request('POST', '/workbench/batches', {'suite': suite})
            self.assertEqual(status, 202, response)
            bid = response['batch']['id']
            detail = self.batch_finish(bid)
            self.assertEqual(detail['batch']['status'], 'completed', detail)
            self.assertEqual(run_suite.call_count, 1)
        self.assertEqual(len(detail['results']), 3)
        cid = detail['results'][0]['case_id']
        status, result = self.request('GET', f'/workbench/batches/{bid}/results/{cid}')
        self.assertEqual(status, 200)
        self.assertTrue(result['wire_events'])
        status, page = self.request('GET', f'/workbench/batches/{bid}/report', raw=True)
        self.assertEqual(status, 200)
        self.assertIn('&lt;script&gt;', page)
        self.assertNotIn('<script>青松</script>', page)
        status, next_batch = self.request('POST', '/workbench/batches', {'suite': self.suite(), 'previous_id': bid})
        self.assertEqual(status, 202, next_batch)
        self.assertEqual(self.batch_finish(next_batch['batch']['id'])['batch']['status'], 'completed')
        self.assertEqual(self.formal_hashes(), self.baseline)

    def test_continue_workspace_keeps_grown_memory_and_isolates_recent(self):
        with patch('continuity.sandbox_summary.SandboxSummary', FakeSummary):
            first = self.session({'mode': 'none', 'enabled': True}, summary=True)
            self.finish(self.start(first, 'PRIVATE_FIRST_TURN'))
            status, result = self.request('POST', '/workbench/sessions', {'continue_from': first})
            self.assertEqual(status, 201, result)
            second = result['session']['id']
            self.assertEqual(result['session']['workspace_id'], first)
            detail = self.request('GET', f'/workbench/sessions/{second}')[1]
            self.assertEqual(detail['messages'], [])
            self.assertIn('GENERATED_ONLY', detail['memory']['text'])
            self.assertIs(self.server.workbench.locks[first], self.server.workbench.locks[second])
            self.finish(self.start(second, 'SECOND_TURN'))
            prompt = self.requests[-1]['body']['input'][0]['content']
            self.assertIn('GENERATED_ONLY', prompt)
            self.assertNotIn('PRIVATE_FIRST_TURN', prompt)
            status, disabled = self.request('POST', '/workbench/sessions', {
                'continue_from': second, 'memory': {'mode': 'none', 'enabled': False},
                'summary': {'enabled': False}})
            self.assertEqual(status, 201, disabled)
            third = disabled['session']['id']
            self.finish(self.start(third))
            self.assertNotIn('GENERATED_ONLY', self.requests[-1]['body']['input'][0]['content'])
            self.assertIn('GENERATED_ONLY', self.request('GET', f'/workbench/sessions/{third}/memory')[1]['text'])
            restarted = workbench.Workbench(self.source, self.managed)
            self.assertIs(restarted.locks[first], restarted.locks[third])
            status, error = self.request('POST', '/workbench/sessions', {
                'continue_from': first, 'memory': {'mode': 'context', 'text': 'different source'}})
            self.assertEqual(status, 409, error)

    def test_continue_enables_retrieval_without_resetting_empty_branch(self):
        with patch('continuity.sandbox_summary.SandboxSummary', FakeSummary), \
                patch.object(evaluation, 'create_snapshot', wraps=evaluation.create_snapshot) as freeze:
            first = self.session({'mode': 'none', 'enabled': False}, retrieval=False, summary=True)
            original = self.server.workbench.sessions[first]
            snapshot_id = original['history_snapshot_id']
            self.assertIsNotNone(snapshot_id)
            config = workbench.restore_config(original['config'])
            with closing(db.connect(config.db_path)) as conn:
                source_max = conn.execute('SELECT MAX(id) FROM chat_log').fetchone()[0]
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM messages').fetchone()[0], 1)
            self.finish(self.start(first, 'LEARNED_PRIVATE_FACT'))
            self.assertNotIn('tools', self.requests[0]['body'])
            first_prompt = self.requests[0]['body']['input'][0]['content']
            self.assertNotIn('FULL_SELECTED', first_prompt)
            self.assertNotIn('FORMAL_RECENT_SECRET', first_prompt)
            before = self.request('GET', f'/workbench/sessions/{first}')[1]
            self.assertGreater(before['messages'][0]['id'], source_max)
            self.assertIn('GENERATED_ONLY', before['memory']['text'])
            with closing(db.connect(config.db_path)) as conn:
                saved = [tuple(r) for r in conn.execute('SELECT id,content FROM chat_log ORDER BY id')]
            status, response = self.request('POST', '/workbench/sessions', {
                'continue_from': first,
                'retrieval': {'enabled': True, 'profile': self.profile('retrieval')},
                'summary': {'enabled': False}})
            self.assertEqual(status, 201, response)
            second = response['session']['id']
            self.assertEqual(response['session']['workspace_id'], original['workspace_id'])
            self.assertEqual(response['session']['history_snapshot_id'], snapshot_id)
            self.assertEqual(self.server.workbench.sessions[second]['config']['db_path'], str(config.db_path))
            detail = self.request('GET', f'/workbench/sessions/{second}')[1]
            self.assertEqual(detail['messages'], [])
            self.assertEqual(detail['memory'], before['memory'])
            with closing(db.connect(config.db_path)) as conn:
                self.assertEqual([tuple(r) for r in conn.execute('SELECT id,content FROM chat_log ORDER BY id')], saved)
            run, events = self.finish(self.start(second))
            self.assertEqual(run['status'], 'completed', events)
            self.assertIn('retrieval_handoff', [e['phase'] for e in events])
            loaded = next(e for e in events if e['phase'] == 'context_loaded')
            self.assertEqual(loaded['data']['stats']['recent_messages'], 0)
            wire = self.request('GET', f'/workbench/runs/{run["id"]}/wire')[1]['wire_events']
            main = next(e for e in wire if e['role'] == 'main')
            self.assertTrue(main['request']['tools'])
            self.assertNotIn('GENERATED_ONLY', main['request']['input'][0]['content'])
            self.assertNotIn('LEARNED_PRIVATE_FACT', main['request']['input'][0]['content'])
            self.assertTrue(any(e['role'] == 'retrieval' for e in wire))
            after = self.request('GET', f'/workbench/sessions/{second}')[1]
            self.assertEqual(after['memory'], before['memory'])
            self.assertGreater(after['messages'][0]['id'], before['messages'][-1]['id'])
            self.assertEqual(FakeSummary.updates, 1)
            self.assertEqual(freeze.call_count, 1)
        self.assertEqual(self.formal_hashes(), self.baseline)

    def test_batch_injection_switch_keeps_frozen_history_and_real_retrieval(self):
        manifest_path = self.source.maint_state_dir / 'users/user/manifest.json'
        manifest = workbench.read(manifest_path)
        summary_path = Path(manifest['summary_path'])
        summary_path.write_text('FULL_SELECTED 机械测试暗号是青松', 'utf-8')
        manifest['summary_sha'] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
        workbench.write(manifest_path, manifest)
        self.baseline = self.formal_hashes()
        suite = self.suite([{'id': 'injected', 'mode': 'pipeline', 'memory_id': 'current', 'inject_memory': True},
                            {'id': 'uninjected', 'mode': 'pipeline', 'memory_id': 'current', 'inject_memory': False}])
        suite['retrieval_profile'] = self.profile('retrieval')
        status, response = self.request('POST', '/workbench/batches', {'suite': suite})
        self.assertEqual(status, 202, response)
        detail = self.batch_finish(response['batch']['id'])
        self.assertEqual(detail['batch']['status'], 'completed', detail)
        self.assertEqual(len(detail['results']), 2)
        snapshot_hashes = set()
        for result in detail['results']:
            self.assertEqual(result['status'], 'passed', result)
            snapshot_hashes.add(result['snapshot_sha256'])
            _, artifact = self.request('GET', '/workbench/batches/' + response['batch']['id']
                                       + '/results/' + result['case_id'])
            injected = result['identity'][2] == 'injected'
            main = next(e for e in artifact['wire_events'] if e['role'] == 'main')
            self.assertEqual('FULL_SELECTED' in main['request']['input'][0]['content'], injected)
            retrieval = [e for e in artifact['wire_events'] if e['role'] == 'retrieval']
            self.assertTrue(retrieval)
            self.assertEqual('FULL_SELECTED' in json.dumps(retrieval, ensure_ascii=False), injected)
            self.assertIn('青松', result['answer'])
            self.assertTrue(result['tool_trace'])
        self.assertEqual(len(snapshot_hashes), 1)
        self.assertEqual(self.formal_hashes(), self.baseline)
        suite['contexts'][0]['inject_memory'] = 'false'
        self.assertEqual(self.request('POST', '/workbench/batches/plan', {'suite': suite})[0], 400)

    def test_summary_batch_hook_real_module_and_private_cases(self):
        with patch('continuity.sandbox_summary.SandboxSummary', FakeSummary):
            suite = self.suite([{'id': 'none', 'mode': 'none'}, {'id': 'context', 'mode': 'context', 'text': 'base'}])
            status, response = self.request('POST', '/workbench/batches', {'suite': suite, 'summary': {'enabled': True}})
            self.assertEqual(status, 202, response)
            detail = self.batch_finish(response['batch']['id'])
            self.assertEqual(detail['batch']['status'], 'completed', detail)
            self.assertEqual(FakeSummary.updates, 2)
            self.assertTrue(all(r['summary']['status'] == 'published' for r in detail['results']))

    def test_reject_host_origin_paths_credentials_and_summary_capabilities(self):
        for headers in ({'Host': 'evil.example'}, {'Origin': 'https://evil.example'}, {'Origin': 'null'},
                        {'Sec-Fetch-Site': 'cross-site'}):
            self.assertEqual(self.request('GET', '/workbench/bootstrap', headers=headers)[0], 403)
        self.assertEqual(self.request('GET', '/workbench/sessions/../../config.json')[0], 400)
        for context in ({'id': 'x', 'mode': 'context', 'file': '/etc/passwd'},
                        {'id': 'x', 'mode': 'snapshot', 'snapshot': '/etc'},
                        {'id': 'x', 'mode': 'snapshot', 'memory_id': '../../etc'}):
            self.assertEqual(self.request('POST', '/workbench/batches/plan', {'suite': self.suite([context])})[0], 400)
        for override in ({'api_key': KEY}, {'base_url': 'https://user:pass@example.org'},
                         {'auth_file': '/tmp/auth.json'}, {'parameters': {'headers': {'X-Key': KEY}}}):
            model = {**self.profile(), **override}
            self.assertEqual(self.request('POST', '/workbench/sessions', {'model': model, 'memory': {'mode': 'none'}})[0], 400)
        for options in ({'enabled': True, 'model': 'unknown-model'}, {'enabled': True, 'max_output_tokens': 100},
                        {'enabled': True, 'cli': '/tmp/x'}, {'enabled': 'yes'}):
            self.assertEqual(self.request('POST', '/workbench/batches/plan', {'suite': self.suite(), 'summary': options})[0], 400)
        self.assertEqual(self.requests, [])

    def test_real_summary_engine_grows_empty_branch_over_http_and_continues(self):
        # Reuse the summary owner's fixture, which replaces urllib inside the
        # maintenance subprocess BEFORE its real final-send tap is installed.
        from test_sandbox_summary import FAKE_CLI, ENGINE
        cli = self.root / 'synthetic-summary-cli.py'
        cli.write_text(FAKE_CLI.replace('ALIAS_PARENT', repr(str(ENGINE / 'review-v2'))))
        self.source.maint_cli = cli
        selected = self.profile('summary-none')
        selected['parameters'] = {'reasoning': {'effort': 'none'}}
        status, created = self.request('POST', '/workbench/sessions', {
            'model': self.profile(), 'memory': {'mode': 'none', 'enabled': True},
            'summary': {'enabled': True, 'model': selected['model'], 'profile': selected}})
        self.assertEqual(status, 201, created)
        sid = created['session']['id']
        run, events = self.finish(self.start(sid, '我喜欢合成测试中的蓝色。'))
        self.assertEqual(run['status'], 'completed', events)
        self.assertIn('maintenance_published', [e['phase'] for e in events], events)
        memory = self.request('GET', f'/workbench/sessions/{sid}/memory')[1]
        self.assertIn('蓝色', memory['text'])
        wire = self.request('GET', f'/workbench/runs/{run["id"]}/wire')[1]['wire_events']
        summary_calls = [e for e in wire if e['role'] == 'summary']
        self.assertTrue(summary_calls)
        self.assertTrue(all(e['api_format'] == 'responses' for e in summary_calls))
        self.assertTrue(all(e['request']['reasoning']['effort'] == 'none' for e in summary_calls))
        self.assertNotIn('PRIVATE-COT', json.dumps(wire))
        self.assertNotIn('synthetic-key-never-persist', json.dumps(wire))
        status, continued = self.request('POST', '/workbench/sessions', {'continue_from': sid})
        self.assertEqual(status, 201, continued)
        new_id = continued['session']['id']
        self.assertEqual(self.request('GET', f'/workbench/sessions/{new_id}')[1]['messages'], [])
        self.finish(self.start(new_id, '我也喜欢合成测试中的绿色。'))
        self.assertIn('蓝色', self.requests[-1]['body']['input'][0]['content'])
        self.assertEqual(self.formal_hashes(), self.baseline)

    def test_real_batch_summary_freezes_structured_seed_and_records_summary_wire(self):
        from test_sandbox_summary import SandboxSummaryTests
        fixture = SandboxSummaryTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        source, initial = fixture.source()
        connection = fixture.connection(source)
        connection.execute("INSERT INTO chat_log(id,created_at,instance,conversation_id,role,content) "
            "VALUES(1,?,?,?,?,?)", (time.time() - 100, source.instance, 'old', 'user', 'Synthetic baseline evidence.'))
        connection.commit()
        connection.execute('PRAGMA journal_mode=DELETE')
        connection.close()
        self.source = source
        self.server.workbench.source = source
        self.baseline = self.formal_hashes()
        suite = self.suite([{'id': 'frozen', 'mode': 'snapshot', 'memory_id': 'current'},
                            {'id': 'static', 'mode': 'context', 'memory_id': 'current'}])
        suite['models'].append(self.profile('another-main'))
        status, response = self.request('POST', '/workbench/batches', {
            'suite': suite, 'summary': {'enabled': True, 'max_requests': 9, 'max_output_tokens': 64000}})
        self.assertEqual(status, 202, response)
        detail = self.batch_finish(response['batch']['id'])
        self.assertEqual(detail['batch']['status'], 'completed', detail)
        self.assertEqual(len(detail['results']), 4)
        for result in detail['results']:
            summary = result['summary']
            self.assertNotEqual(summary['status'], 'failed', summary)
            self.assertEqual(summary['seed_mode'], 'cloned_current_structured_state')
            self.assertEqual(summary['before_text'], initial.text)
            status, artifact = self.request('GET', '/workbench/batches/' + response['batch']['id']
                                            + '/results/' + result['case_id'])
            self.assertEqual(status, 200)
            summary_calls = [e for e in artifact['wire_events'] if e['role'] == 'summary']
            self.assertTrue(summary_calls)
            self.assertNotIn('PRIVATE-COT', json.dumps(artifact))
            self.assertNotIn('synthetic-key-never-persist', json.dumps(artifact))
        seeds = list((self.managed / 'batch-inputs' / response['batch']['id']).glob('*/state/state.sqlite3'))
        self.assertEqual(len(seeds), 1)
        self.assertEqual(self.formal_hashes(), self.baseline)

    def test_current_static_session_keeps_original_rows_and_ids_with_retrieval_off(self):
        from test_sandbox_summary import SandboxSummaryTests
        fixture = SandboxSummaryTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        source, initial = fixture.source()
        connection = fixture.connection(source)
        for row_id, role, text in [(1, 'user', 'Synthetic baseline evidence.'), (2, 'assistant', 'Original assistant.')]:
            connection.execute('INSERT INTO chat_log(id,created_at,instance,conversation_id,role,content) '
                'VALUES(?,?,?,?,?,?)', (row_id, time.time() - 100, source.instance, 'original', role, text))
        connection.commit()
        source_max = connection.execute('SELECT MAX(id) FROM chat_log').fetchone()[0]
        connection.execute('PRAGMA journal_mode=DELETE')
        connection.close()
        self.source = source
        self.server.workbench.source = source
        self.baseline = self.formal_hashes()
        sid = self.session({'mode': 'context', 'id': 'current'}, summary=True)
        self.assertIsNotNone(self.server.workbench.sessions[sid]['history_snapshot_id'])
        run, events = self.finish(self.start(sid, '我喜欢合成测试中的蓝色。'))
        self.assertEqual(run['status'], 'completed', events)
        self.assertIn('maintenance_published', [e['phase'] for e in events], events)
        detail = self.request('GET', f'/workbench/sessions/{sid}')[1]
        self.assertGreater(detail['messages'][0]['id'], source_max)
        config = workbench.restore_config(self.server.workbench.sessions[sid]['config'])
        with closing(db.connect(config.db_path)) as private:
            originals = private.execute('SELECT id,content FROM chat_log WHERE conversation_id=? ORDER BY id',
                                        ('original',)).fetchall()
        self.assertEqual([(r['id'], r['content']) for r in originals],
                         [(1, 'Synthetic baseline evidence.'), (2, 'Original assistant.')])
        self.assertNotIn('Original assistant.', self.requests[0]['body']['input'][0]['content'])
        self.assertNotIn('tools', self.requests[0]['body'])
        self.assertIn('蓝色', detail['memory']['text'])
        self.assertEqual(self.formal_hashes(), self.baseline)

    def test_factory_refuses_external_host_and_formal_overlap(self):
        with self.assertRaises(ValueError):
            workbench.serve_workbench(self.source, host='0.0.0.0', port=0, root=self.managed)
        with self.assertRaises(ValueError):
            workbench.Workbench(self.source, self.source.maint_state_dir)


if __name__ == '__main__':
    unittest.main()
