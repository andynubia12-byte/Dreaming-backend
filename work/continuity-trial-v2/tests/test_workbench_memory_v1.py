"""Real loopback HTTP + real v1 SQLite engine, entirely synthetic model transport."""
from __future__ import annotations

import copy
from contextlib import closing
from datetime import datetime, timezone
import http.client
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import uuid
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from continuity import db, workbench
from continuity.config import Config
from continuity.deepseek import DeepSeekClient, Usage
from continuity.providers import normalize_response

KEY = 'synthetic-workbench-v1-key'


def iso(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def native(text='', calls=()):
    output = [{'type': 'function_call', 'id': 'fc', 'call_id': 'call',
               'name': name, 'arguments': json.dumps(args), 'status': 'completed'} for name, args in calls]
    if text:
        output.append({'type': 'message', 'role': 'assistant', 'status': 'completed',
            'content': [{'type': 'output_text', 'text': text}]})
    return {'id': 'synthetic', 'status': 'completed', 'output': output,
            'usage': {'input_tokens': 10, 'output_tokens': 10}}


class FakeMain:
    message = staticmethod(DeepSeekClient.message)
    content = staticmethod(DeepSeekClient.content)

    def __init__(self, captured):
        self.captured, self.usage = captured, Usage()

    def chat_stream(self, messages, *, on_event, **options):
        self.captured.append(copy.deepcopy(messages))
        on_event('content', text='这是已保存的合成回答。')
        return normalize_response(native('这是已保存的合成回答。'))


class WorkbenchMemoryV1Tests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        formal = self.root / 'formal'
        self.source = Config(project_root=formal, db_path=formal / 'chat.sqlite',
            maint_state_dir=formal / 'state', runs_dir=formal / 'runs',
            usage_file=formal / 'usage.json', user_id='synthetic', instance='synthetic',
            api_key_env='V1_TEST_KEY')
        self.source.ensure_dirs()
        self.base = time.time() - 10000
        with closing(db.connect(self.source.db_path)) as conn:
            db.init_db(conn)
            conn.executemany('INSERT INTO messages(conversation_id,node_id,message_id,role,text,'
                'create_time,is_selected_branch,is_assistant_reasoning) VALUES(?,?,?,?,?,?,1,0)', [
                ('fixture', 'first', 'first', 'user', '我喜欢散步。', self.base),
                ('fixture', 'second', 'second', 'user', '我喜欢阅读。', self.base + 100),
                ('fixture', 'assistant', 'assistant', 'assistant', '助手猜测。', self.base)])
            conn.commit()
            conn.execute('PRAGMA journal_mode=DELETE')
        self.original = {str(p): p.read_bytes() for p in formal.rglob('*') if p.is_file()}
        self.managed = self.root / 'workbench'
        self.calls, self.main_inputs, self.client_configs = [], [], []
        self.fail_model, self.empty = False, False
        env = patch.dict(os.environ, {'V1_TEST_KEY': KEY})
        env.start()
        self.addCleanup(env.stop)
        transport = patch('urllib.request.urlopen', side_effect=self.transport)
        transport.start()
        self.addCleanup(transport.stop)
        create = workbench.create_client

        def client(config, profile, *, recorder=None, role='main'):
            self.client_configs.append((role, config, copy.deepcopy(profile)))
            return FakeMain(self.main_inputs) if role == 'main' else create(
                config, profile, recorder=recorder, role=role)

        factory = patch.object(workbench, 'create_client', side_effect=client)
        factory.start()
        self.addCleanup(factory.stop)
        self.server = workbench.serve_workbench(self.source, root=self.managed, port=0)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()

        def stop():
            self.server.shutdown()
            self.server.server_close()
            thread.join(2)
        self.addCleanup(stop)

    def transport(self, request, **options):
        body = json.loads(request.data)
        self.calls.append(body)
        if self.fail_model:
            raise RuntimeError(KEY)
        history = body['input']
        text = '\n'.join(item['content'] if isinstance(item.get('content'), str) else
                         '\n'.join(part['text'] for part in item.get('content', []) if 'text' in part)
                         for item in history if item.get('role') == 'user')
        if body.get('tools'):
            payload = native('证据已取得。') if any(i.get('type') == 'function_call_output' for i in history) else native(
                calls=[('read_message', {'source_id': 'msg:fixture#first'})])
        elif body.get('text', {}).get('format', {}).get('type') == 'json_object':
            evidence = [json.loads(line) for line in text.split('VERIFIED USER EVIDENCE:\n')[1].splitlines()]
            ids = [item['id'] for item in evidence]
            section = '' if self.empty else ('### 更新 ' + text.split('SIMULATED NOW: ')[1][:10] + '\n' +
                '\n'.join(item['text'] + ' (' + item['id'] + ')' for item in evidence))
            payload = native(json.dumps(dict(section_markdown=section, summary='合成摘要',
                sources=[] if self.empty else ids, supersedes=[], confidence=0.9)))
        else:
            payload = native('## A. 稳定背景\n- 喜欢散步。 (msg:fixture#first)\n'
                '## B. 交流偏好\n## C. 有时间依据的当前状态\n## D. 未找到 / 不确定\n- 未找到其他信息。')
        response = io.BytesIO(json.dumps(payload).encode())
        response.status = 200
        return response

    def request(self, method, path, body=None, *, raw=False):
        conn = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=10)
        conn.request(method, path, body=json.dumps(body) if body is not None else None,
                     headers={'Content-Type': 'application/json'})
        response = conn.getresponse()
        data = response.read().decode()
        status = response.status
        conn.close()
        return status, data if raw else json.loads(data)

    def profile(self, role):
        return {'id': role, 'model': 'synthetic-' + role, 'api_format': 'responses',
            'base_url': 'https://synthetic.invalid', 'api_key_env': 'V1_TEST_KEY',
            'parameters': {'reasoning': {'effort': 'none'}}}

    def session(self, *, enabled=False, budget=20):
        status, value = self.request('POST', '/workbench/sessions', {
            'model': self.profile('main'), 'memory': {'mode': 'none'},
            'retrieval': {'enabled': False}, 'summary': {'engine': 'v1', 'enabled': enabled,
                'profile': self.profile('summary'), 'max_requests': budget, 'max_output_tokens': 16000}})
        self.assertEqual(status, 201, value)
        return value['session']['id']

    def job_body(self, operation='initialize', version=0, **extra):
        body = {'request_id': str(uuid.uuid4()), 'operation': operation, 'expected_version': version}
        if operation == 'initialize':
            body.update(now=iso(self.base + 1000), batches=[{'now': iso(self.base + 500),
                'source_ids': ['msg:fixture#first']}])
        elif operation == 'append':
            body.update(now=iso(self.base + 2000), source_ids=['msg:fixture#second'])
        return {**body, **extra}

    def start(self, sid, body):
        status, value = self.request('POST', f'/workbench/sessions/{sid}/memory/v1/jobs', body)
        self.assertEqual(status, 202, value)
        return value

    def finish(self, rid):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status, value = self.request('GET', '/workbench/runs/' + rid)
            if value['run']['status'] not in ('accepted', 'running'):
                _, sse = self.request('GET', f'/workbench/runs/{rid}/events', raw=True)
                events = [json.loads(line[6:]) for line in sse.splitlines() if line.startswith('data: ')]
                self.assertEqual(events[-1]['phase'], 'run_complete')
                return value['run'], events
            time.sleep(.01)
        self.fail('worker did not finish')

    def current(self, sid):
        status, current = self.request('GET', f'/workbench/sessions/{sid}/memory/v1')
        self.assertEqual(status, 200, current)
        return current

    def chat(self, sid, text, request_id=None):
        status, value = self.request('POST', f'/workbench/sessions/{sid}/chat', {
            'request_id': request_id or str(uuid.uuid4()), 'message': text})
        self.assertEqual(status, 202, value)
        run, events = self.finish(value['run_id'])
        self.assertEqual(run['status'], 'completed', run)
        return run, events

    def test_http_lifecycle_chat_injection_shared_memory_and_private_recent(self):
        sid = self.session()
        self.assertEqual(self.current(sid)['version'], 0)
        run, _ = self.finish(self.start(sid, self.job_body())['run_id'])
        self.assertEqual(run['result']['status'], 'published')
        self.assertEqual(run['result']['version'], 1)
        run, _ = self.finish(self.start(sid, self.job_body('append', 1))['run_id'])
        self.assertEqual(run['result']['version'], 2)
        current = self.current(sid)
        self.assertEqual(current['stable_text'], '')
        self.assertEqual(self.request('GET', f'/workbench/sessions/{sid}/memory/v1/versions/2')[1], current)
        versions = self.request('GET', f'/workbench/sessions/{sid}/memory/v1/versions')[1]['versions']
        self.assertEqual([v['version'] for v in versions], [0, 1, 2])
        self.assertFalse(self.request('GET', f'/workbench/sessions/{sid}/memory/v1/versions/0')[1]['present'])
        self.chat(sid, 'ONLY_RECENT_A')
        self.assertIn(current['text'], self.main_inputs[-1][0]['content'])
        self.assertEqual(self.current(sid), current)  # disabled still injects, no automatic write
        status, created = self.request('POST', '/workbench/sessions', {
            'continue_from': sid, 'summary': {'enabled': True}})
        self.assertEqual(status, 201, created)
        other = created['session']['id']
        self.assertEqual(created['session']['workspace_id'], sid)
        self.assertEqual(self.request('GET', f'/workbench/sessions/{other}')[1]['messages'], [])
        run, events = self.chat(other, '我今天学了游泳。')
        self.assertNotIn('ONLY_RECENT_A', self.main_inputs[-1][0]['content'])
        self.assertEqual(run['summary']['status'], 'published')
        self.assertIn('maintenance_published', [e['phase'] for e in events])
        self.assertEqual(self.current(sid)['version'], 3)
        self.assertEqual(self.current(sid)['stable_text'], '')
        request = workbench.read(self.managed / 'runs' / run['id'] / 'summary-request.json')
        self.assertEqual(len(request['source_ids']), 1)
        self.assertRegex(request['source_ids'][0], r'^chat#\d+$')
        roles = self.client_configs
        summary = [(cfg, p) for role, cfg, p in roles if role == 'summary']
        self.assertTrue(all(cfg.max_requests == cfg.trial_max_requests == 20 for cfg, _ in summary))
        self.assertTrue(all(p['model'] == 'synthetic-summary' and p['parameters']['reasoning']['effort'] == 'none'
                            and p['parameters']['max_output_tokens'] == 16000 for _, p in summary))
        self.assertTrue(all(cfg.usage_file.name == 'memory-v1-usage.json' for cfg, _ in summary))
        self.assertNotEqual(summary[0][0].usage_file, next(cfg.usage_file for role, cfg, _ in roles if role == 'main'))
        self.assertEqual(workbench.read(summary[0][0].usage_file)['requests'], 3)
        self.assertEqual(self.current(self.session())['version'], 0)
        status, muted = self.request('POST', '/workbench/sessions', {'continue_from': sid,
            'summary': {'enabled': False}, 'memory': {'mode': 'none', 'enabled': False}})
        self.assertEqual(status, 201, muted)
        self.chat(muted['session']['id'], '无记忆注入。')
        self.assertNotIn(self.current(sid)['text'], self.main_inputs[-1][0]['content'])
        self.assertEqual({str(p): p.read_bytes() for p in self.source.project_root.rglob('*') if p.is_file()},
                         self.original)

    def test_idempotency_noop_and_request_namespace(self):
        sid = self.session()
        body = self.job_body()
        accepted = self.start(sid, body)
        self.finish(accepted['run_id'])
        reused = self.start(sid, body)
        self.assertTrue(reused['reused'])
        self.assertEqual(reused['run_id'], accepted['run_id'])
        endpoint = f'/workbench/sessions/{sid}/memory/v1/jobs'
        self.assertEqual(self.request('POST', endpoint, {**body, 'expected_version': 1})[0], 409)
        self.assertEqual(self.request('POST', f'/workbench/sessions/{sid}/chat', {
            'request_id': body['request_id'], 'message': '不能混用。'})[0], 409)
        request_id = str(uuid.uuid4())
        self.chat(sid, '普通聊天。', request_id)
        self.assertEqual(self.request('POST', endpoint, self.job_body('append', 1, request_id=request_id))[0], 409)
        calls = len(self.calls)
        run, events = self.finish(self.start(sid, self.job_body('append', 1,
            source_ids=['msg:fixture#first']))['run_id'])
        self.assertEqual(run['result']['status'], 'noop')
        self.assertEqual(self.current(sid)['version'], 1)
        self.assertEqual(len(self.calls), calls)
        self.assertIn('maintenance_noop', [e['phase'] for e in events])
        self.assertNotIn('maintenance_published', [e['phase'] for e in events])
        self.empty = True
        _, sibling = self.request('POST', '/workbench/sessions', {
            'continue_from': sid, 'summary': {'enabled': True}})
        run, events = self.chat(sibling['session']['id'], '这轮没有长期信息。')
        self.assertEqual(run['summary']['status'], 'noop')
        self.assertIn('maintenance_noop', [e['phase'] for e in events])
        self.assertNotIn('maintenance_published', [e['phase'] for e in events])

    def test_failure_restart_retains_version_and_recovers_committed_job_without_calls(self):
        sid = self.session()
        first = self.start(sid, self.job_body())
        self.finish(first['run_id'])
        previous = self.current(sid)
        self.fail_model = True
        failed = self.start(sid, self.job_body('append', 1))
        run, _ = self.finish(failed['run_id'])
        self.assertEqual(run['status'], 'failed')
        self.assertEqual(run['error_type'], 'MemoryV1Error')
        self.assertIn('model request failed', run['error'])
        self.assertNotIn(KEY, json.dumps(run))
        self.assertEqual(self.current(sid), previous)
        for rid in (first['run_id'], failed['run_id']):
            path = self.managed / 'runs' / rid / 'run.json'
            saved = workbench.read(path)
            saved.update(status='running')  # crash after SQLite commit / before final run persistence
            saved.pop('result', None)
            workbench.write(path, saved)
        with patch.object(workbench, 'create_client', side_effect=AssertionError('boot must not dispatch')):
            restarted = workbench.Workbench(self.source, self.managed)
        recovered = restarted.runs[first['run_id']]
        self.assertEqual(recovered['result']['memory'], previous)
        self.assertEqual(recovered['result']['status'], 'published')
        self.assertTrue(recovered['recovered'])
        interrupted = restarted.runs[failed['run_id']]
        self.assertEqual(interrupted['status'], 'interrupted')
        self.assertEqual(interrupted['error_type'], 'InterruptedMemoryJob')
        self.assertIn('新 request_id', interrupted['error'])
        self.assertEqual(restarted.memory_v1.get(sid), previous)
        self.assertTrue(restarted.memory_v1.start(sid, self.job_body('append', 1,
            request_id=failed['run']['request_id']))['reused'])
        self.fail_model = False
        self.finish(self.start(sid, self.job_body('append', 1))['run_id'])
        self.assertEqual(self.current(sid)['version'], 2)

    def test_zero_and_cumulative_summary_budget_keep_chat_answer(self):
        zero = self.session(budget=0)
        run, _ = self.finish(self.start(zero, self.job_body())['run_id'])
        self.assertEqual(run['status'], 'failed')
        self.assertEqual(run['error_type'], 'BudgetExceeded')
        self.assertIn('summary.max_requests=0', run['error'])
        self.assertEqual(self.calls, [])
        sid = self.session(enabled=True, budget=1)
        self.finish(self.start(sid, self.job_body())['run_id'])
        previous = self.current(sid)
        run, events = self.chat(sid, '摘要预算已用完，但保留聊天答案。')
        self.assertEqual(run['summary']['status'], 'failed')
        self.assertEqual(self.current(sid), previous)
        self.assertTrue((self.managed / 'runs' / run['id'] / 'answer.json').is_file())
        self.assertIn('maintenance_failed', [e['phase'] for e in events])
        self.assertEqual(len(self.calls), 1)

    def test_committed_job_and_auto_append_survive_summary_artifact_and_observer_errors(self):
        sid = self.session(enabled=True)
        body = self.job_body()
        write = workbench.write
        emit = workbench.Observer.emit

        def broken_summary(path, value):
            if path.name == 'summary.json':
                raise OSError(KEY)
            return write(path, value)

        def broken_publication(observer, phase, **data):
            if phase == 'maintenance_published':
                raise RuntimeError(KEY)
            return emit(observer, phase, **data)

        with patch.object(workbench, 'write', side_effect=broken_summary), patch.object(
                workbench.Observer, 'emit', new=broken_publication):
            accepted = self.start(sid, body)
            job, job_events = self.finish(accepted['run_id'])
            self.assertEqual(job['status'], 'completed', job)
            self.assertEqual(job['result']['memory'], self.current(sid))
            self.assertEqual(job['result']['version'], 1)
            chat_run, chat_events = self.chat(sid, '已提交后记录异常也不能谎报生成失败。')
            self.assertEqual(chat_run['summary']['memory'], self.current(sid))
            self.assertEqual(chat_run['summary']['version'], 2)
        for run, events, field in ((job, job_events, 'result'), (chat_run, chat_events, 'summary')):
            folder = self.managed / 'runs' / run['id']
            saved = workbench.read(folder / 'run.json')
            self.assertEqual(saved['status'], 'completed')
            self.assertEqual(saved[field]['status'], 'published')
            self.assertEqual({w['stage'] for w in saved['summary_warnings']},
                             {'summary_artifact', 'summary_observer'})
            self.assertNotIn('error', saved)
            self.assertNotIn(KEY, json.dumps(saved))
            self.assertFalse((folder / 'summary.json').exists())
            self.assertNotIn('maintenance_failed', [e['phase'] for e in events])
            self.assertEqual(events[-1]['data']['status'], 'completed')
        self.assertTrue(self.start(sid, body)['reused'])
        self.assertEqual(len(self.calls), 2)

    def test_legacy_empty_session_initializes_real_summary_without_model_calls(self):
        from continuity.sandbox_summary import SandboxSummary
        with closing(db.connect(self.source.db_path)) as conn:
            conn.execute('DELETE FROM messages')
            conn.execute('DELETE FROM chat_log')
            conn.commit()
            conn.execute('PRAGMA journal_mode=DELETE')
        summary = {'enabled': True, 'profile': self.profile('summary'), 'max_requests': 2}
        wb = self.server.workbench
        with patch.object(workbench, 'create_client', side_effect=AssertionError('no model calls')), patch(
                'urllib.request.urlopen', side_effect=AssertionError('no network')):
            status, created = self.request('POST', '/workbench/sessions', {
                'model': self.profile('main'), 'memory': {'mode': 'none'},
                'retrieval': {'enabled': False}, 'summary': summary})
            self.assertEqual(status, 201, created)
            session = wb.session(created['session']['id'])
            config = workbench.restore_config(session['config'])
            engine = wb.engine(config, session)
            self.assertIsInstance(engine, SandboxSummary)
            self.assertEqual(engine.initialize()['status'], 'ready')
            self.assertEqual(engine.status()['source_mode'], 'none')
            self.assertFalse(engine.published().present)
            suite = {'models': [self.profile('main')], 'questions': [{'id': 'q', 'text': '合成问题'}],
                'contexts': [{'id': 'none', 'mode': 'none'}],
                'budgets': {'max_requests': 2, 'trial_max_requests': 2, 'max_input_tokens': 10000}}
            status, plan = self.request('POST', '/workbench/batches/plan', {'suite': suite, 'summary': summary})
            self.assertEqual(status, 200, plan)
            self.assertTrue(plan['valid'])
            self.assertEqual(plan['summary']['engine'], 'v2.6')
            # Batch seeds have no workspace.json; they use the same strict legacy boundary.
            seed = workbench.isolated(self.source, self.managed / 'batch-inputs' / 'legacy-empty',
                                      self.profile('main'), wb.retrieval_profile)
            seed_engine = wb.engine(seed, {'initial': session['initial'],
                'summary': wb.runtime_summary(summary, [])})
            self.assertIsInstance(seed_engine, SandboxSummary)
            self.assertEqual(seed_engine.initialize()['status'], 'ready')
        self.assertEqual(self.calls, [])

    def test_shared_workspace_lock_and_auto_append_crash_recovery(self):
        sid = self.session(enabled=True)
        status, sibling = self.request('POST', '/workbench/sessions', {'continue_from': sid})
        self.assertEqual(status, 201, sibling)
        other = sibling['session']['id']
        body = self.job_body()
        entered, release = threading.Event(), threading.Event()

        def delayed(request, **options):
            entered.set()
            release.wait(5)
            return self.transport(request, **options)

        with patch('urllib.request.urlopen', side_effect=delayed):
            try:
                accepted = self.start(sid, body)
                self.assertTrue(entered.wait(2))
                self.assertTrue((self.managed / 'runs' / accepted['run_id'] / 'sources.sqlite').is_file())
                self.assertEqual(self.current(other)['version'], 0)  # no SQLite lock spans the API call
                self.assertTrue(self.start(other, body)['reused'])
                self.assertEqual(self.request('POST', f'/workbench/sessions/{other}/memory/v1/jobs',
                    self.job_body())[0], 409)
                self.assertEqual(self.request('POST', f'/workbench/sessions/{other}/chat', {
                    'request_id': str(uuid.uuid4()), 'message': '工作区忙时拒绝。'})[0], 409)
            finally:
                release.set()
            self.finish(accepted['run_id'])
        run, _ = self.chat(other, '自动追加已提交。')
        self.assertEqual(run['summary']['status'], 'published')
        folder = self.managed / 'runs' / run['id']
        saved = workbench.read(folder / 'run.json')
        saved.update(status='running')
        saved.pop('summary')
        workbench.write(folder / 'run.json', saved)
        # Exercise both crash windows: after core commit and after terminal run persistence.
        for rid in (accepted['run_id'], run['id']):
            event_path = self.managed / 'runs' / rid / 'events.jsonl'
            events = workbench.jsonlines(event_path)
            event_path.write_text(''.join(json.dumps(e) + '\n' for e in events if e['phase'] != 'run_complete'))
        with patch.object(workbench, 'create_client', side_effect=AssertionError('no boot requests')):
            recovered = workbench.Workbench(self.source, self.managed)
        self.assertEqual(recovered.runs[run['id']]['status'], 'completed')
        self.assertEqual(recovered.runs[run['id']]['summary'], run['summary'])
        for rid in (accepted['run_id'], run['id']):
            self.assertEqual(workbench.jsonlines(self.managed / 'runs' / rid / 'events.jsonl')[-1]['phase'],
                             'run_complete')

    def test_explicit_stable_refresh_and_preflight_rejections(self):
        sid = self.session()
        self.finish(self.start(sid, self.job_body())['run_id'])
        endpoint = f'/workbench/sessions/{sid}/memory/v1/jobs'
        for body in (self.job_body('append', 1, source_ids=['msg:fixture#assistant']),
                     self.job_body('append', 1, source_ids=[]),
                     self.job_body('refresh_stable', 1, source_ids=[]),
                     self.job_body('append', 1, now=iso(time.time() + 10000))):
            self.assertEqual(self.request('POST', endpoint, body)[0], 400)
        self.assertEqual(self.request('POST', endpoint, self.job_body('append', 0))[0], 409)
        before = self.current(sid)
        run, events = self.finish(self.start(sid, self.job_body('refresh_stable', 1))['run_id'])
        self.assertEqual(run['status'], 'completed', run)
        stable = self.current(sid)['stable_text']
        self.assertIn('喜欢散步', stable)
        self.assertEqual(self.current(sid)['event_text'], before['event_text'])
        self.assertIn('stable_tool_result', [e['phase'] for e in events])
        status, sibling = self.request('POST', '/workbench/sessions', {
            'continue_from': sid, 'summary': {'enabled': True}})
        self.assertEqual(status, 201, sibling)
        chat_run, _ = self.chat(sibling['session']['id'], '再补充一条事件。')
        self.assertEqual(chat_run['summary']['status'], 'published')
        self.assertEqual(self.current(sid)['stable_text'], stable)
        self.assertEqual(self.request('POST', '/workbench/sessions', {
            'continue_from': sid, 'summary': {'engine': 'v2.6'}})[0], 409)
        self.assertEqual(self.request('POST', '/workbench/sessions', {'model': self.profile('main'),
            'memory': {'mode': 'context', 'text': '旧种子'},
            'summary': {'engine': 'v1', 'enabled': False}})[0], 400)
        self.assertEqual(self.request('POST', '/workbench/batches/plan', {
            'suite': {}, 'summary': {'engine': 'v1', 'enabled': False}})[0], 400)
        status, legacy = self.request('POST', '/workbench/sessions', {'model': self.profile('main'),
            'memory': {'mode': 'none'}, 'summary': {'enabled': False}})
        self.assertEqual(status, 201, legacy)
        old = legacy['session']['id']
        self.assertEqual(self.request('GET', f'/workbench/sessions/{old}/memory/v1')[0], 409)
        self.assertEqual(self.request('POST', f'/workbench/sessions/{old}/memory/v1/jobs', self.job_body())[0], 409)


if __name__ == '__main__':
    unittest.main()
