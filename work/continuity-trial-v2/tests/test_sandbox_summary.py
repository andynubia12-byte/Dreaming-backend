"""Synthetic structured state + admitted engine over the shared Responses client."""
from __future__ import annotations

from contextlib import closing
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True  # Never write caches into the frozen dependency.
sys.path.insert(0, str(ROOT / 'src'))
ENGINE = ROOT.parent / 'memory-maintenance-v1'
sys.path.insert(0, str(ENGINE / 'repair-resolve-v26/src'))
from memmaint.store import Store
from continuity import db
from continuity.config import Config
from continuity.events import EventHub, Observer, observe
from continuity.sandbox_summary import SandboxSummary
from continuity.summary_store import Published, read_published

NOW = 1_700_000_000

# This fixture imports the real admitted CLI, with urllib replaced BEFORE the
# shared provider is invoked. Even a broken test cannot hit a paid endpoint.
FAKE_CLI = '''
import io, json, os, sys, urllib.error, urllib.request
from pathlib import Path
sys.path.insert(0, ALIAS_PARENT)
from run_update_with_citation_alias import install, main
os.environ['DEEPSEEK_API_KEY'] = 'synthetic-key-never-persist'

def urlopen(request, *args, **kwargs):
    body = json.loads(request.data)
    assert request.full_url.endswith('/responses')
    assert 'messages' not in body and 'previous_response_id' not in body
    payload = json.loads(body['input'][1]['content'])
    flag = Path(__file__).with_suffix('.fail')
    if flag.exists():
        error = json.dumps({'error': {'message': 'synthetic-key-never-persist',
                                     'reasoning_content': 'PRIVATE-COT'}}).encode()
        raise urllib.error.HTTPError('https://fixture.invalid', 500, 'fixture', {}, io.BytesIO(error))
    if 'new_messages' in payload:
        msg = next(m for m in payload['new_messages'] if m['role'] == 'user')
        quote = msg['text']
        operation = {'op': 'add', 'kind': 'preference', 'text': quote,
                     'evidence_message_ids': [msg['id']],
                     'quotes': [{'message_id': msg['id'], 'quote': quote}]}
        if 'previous_attempt' not in payload and Path(__file__).with_suffix('.repair').exists():
            operation['quotes'][0]['quote'] = 'invented nonverbatim evidence'
        content = json.dumps({'operations': [operation]})
    elif 'claims' in payload:
        content = json.dumps({'verdicts': [dict(index=c['index'], verdict='supported',
                            role_ok=True, time_ok=True, overgeneralized=False,
                            issue='', corrected_text=None) for c in payload['claims']]})
    else:
        aliases = payload.get('reference_aliases', {})
        content = '# Synthetic published summary\\n\\n' + payload['memory_material']
        # The admitted wire requires aliases on the real summary transport.
        for alias, canonical in sorted(aliases.items(), key=lambda pair: -len(pair[1])):
            content = content.replace('[' + canonical + ']', '[' + alias + ']')
    response = {'id': 'resp_fixture', 'status': 'completed', 'output': [
                   {'type': 'reasoning', 'content': [{'type': 'reasoning_text',
                     'text': 'PRIVATE-COT synthetic-key-never-persist'}]},
                   {'type': 'message', 'role': 'assistant', 'content': [
                     {'type': 'output_text', 'text': content}]}],
                'usage': {'input_tokens': 17, 'output_tokens': 11}}
    stream = io.BytesIO(json.dumps(response).encode())
    stream.status = 200
    return stream

urllib.request.urlopen = urlopen
'''


def hashes(folder):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob('*') if p.is_file() and p.suffix not in ('-wal', '-shm')}


class SandboxSummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.cli = self.root / 'fixture_cli.py'
        self.cli.write_text(FAKE_CLI.replace('ALIAS_PARENT', repr(str(ENGINE / 'review-v2'))))

    def config(self, name='session'):
        root = self.root / name
        root.mkdir(exist_ok=True)
        return Config(project_root=root, db_path=root / 'chat.sqlite', runs_dir=root / 'runs',
                      usage_file=root / 'usage.json', maint_state_dir=root / 'state',
                      maint_cli=self.cli, user_id='fixture-user', instance='fixture',
                      maint_timeout_seconds=20)

    def connection(self, config):
        conn = db.connect(config.db_path)
        self.addCleanup(conn.close)
        db.init_db(conn)
        return conn

    def turn(self, conn, preference='我喜欢合成测试中的蓝色。'):
        ids = []
        for role, text in [('user', preference), ('assistant', 'Synthetic answer retained.')]:
            ids.append(conn.execute('INSERT INTO chat_log(created_at,instance,conversation_id,role,content) '
                                    'VALUES(?,?,?,?,?)', (NOW, 'fixture', 'session', role, text)).lastrowid)
        conn.commit()
        return ids

    def helper(self, config=None, **kwargs):
        return SandboxSummary(config or self.config(), options={'enabled': True, 'max_requests': 9}, **kwargs)

    def source(self):
        config = self.config('source')
        store = Store(config.maint_state_dir)
        try:
            store.ensure_user(config.user_id)
            with store.transaction():
                store.add_message(config.user_id, {'id': 'chat#1', 'role': 'user',
                                  'conversation_id': 'old', 'message_at': '2023-01-01T00:00:00+08:00',
                                  'text': 'Synthetic baseline evidence.'}, 'baseline')
                store.insert_claim(config.user_id, {'text': 'Synthetic baseline claim.',
                                   'kind': 'user_fact', 'status': 'active'}, [], 1)
                version = store.bump_version(config.user_id)
                store.finalize_version(config.user_id, version, summary_text='Synthetic baseline summary.\n',
                                       memory_text='Synthetic baseline material.\n',
                                       evidence_count=1, claims_active=1)
            store.publish(config.user_id)
        finally:
            store.close()
        ledger = config.maint_state_dir / 'ledger'
        ledger.mkdir()
        (ledger / 'usage.json').write_text(json.dumps(dict(requests=298, input_tokens=1000,
                                                        output_tokens=500, errors=2)))
        # Old bulky immutable versions are unnecessary to continue the engine.
        stale = config.maint_state_dir / 'users' / 'stale' / 'versions' / 'old'
        stale.mkdir(parents=True)
        (stale / 'unused.bin').write_bytes(b'unneeded old artifact')
        return config, read_published(config.maint_state_dir, config.user_id)

    def test_disabled_does_not_initialize_or_invoke_engine(self):
        config = self.config()
        original = Published(True, 65, 'Fixed synthetic reference.')
        summary = SandboxSummary(config, initial=original, options={'enabled': False})
        with patch('continuity.sandbox_summary.subprocess.run', side_effect=AssertionError('engine called')):
            self.assertEqual(summary.initialize()['status'], 'disabled')
            self.assertEqual(summary.update(None, 'invalid', 2)['status'], 'disabled')
            self.assertEqual(summary.published(), original)
        self.assertFalse(config.maint_state_dir.exists())

    def test_current_clone_is_readonly_minimal_rebased_and_baseline_not_new_usage(self):
        source, initial = self.source()
        before = hashes(source.maint_state_dir)
        config = self.config()
        summary = self.helper(config, source_config=source, initial=initial)
        metadata = summary.initialize()
        self.assertEqual(metadata['seed_mode'], 'cloned_current_structured_state')
        self.assertEqual(summary.published().text, initial.text)
        self.assertEqual(metadata['usage']['baseline']['requests'], 298)
        self.assertEqual(metadata['usage']['delta']['requests'], 0)
        self.assertEqual(metadata['usage']['cap'], 307)
        self.assertEqual(before, hashes(source.maint_state_dir))
        self.assertFalse(list(config.maint_state_dir.rglob('unused.bin')))
        manifests = list(config.maint_state_dir.glob('users/*/manifest.json'))
        self.assertEqual(len(manifests), 1)
        manifest = json.loads(manifests[0].read_text())
        for name in ('summary', 'memory', 'evidence'):
            path = Path(manifest[name + '_path'])
            self.assertTrue(path.is_relative_to(config.maint_state_dir))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), manifest[name + '_sha'])
        with closing(sqlite3.connect(config.maint_state_dir / 'state.sqlite3')) as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM claims').fetchone()[0], 1)
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM messages').fetchone()[0], 1)
        self.assertFalse(list(config.maint_state_dir.rglob('*.wire.jsonl')))

    def test_real_cli_publishes_new_memory_records_final_wire_and_live_phases(self):
        config = self.config()
        conn = self.connection(config)
        ids = self.turn(conn)
        summary = self.helper(config)
        summary.initialize()
        hub = EventHub()
        events = []
        with observe(Observer(hub, session_id='test')):
            result = summary.update(conn, f'chat#{ids[0]}', str(ids[1]), recorder=events.append)
        self.assertEqual(result['status'], 'ok', result)
        self.assertIn('我喜欢合成测试中的蓝色', summary.published().text)
        self.assertEqual(result['usage']['model_calls'], 3)
        self.assertEqual(len(events), 3)
        for event in events:
            self.assertEqual(event['url'], 'https://api.deepseek.com/responses')
            self.assertEqual(event['api_format'], 'responses')
            self.assertEqual(event['model'], event['request']['model'])
            self.assertGreaterEqual(event['elapsed_seconds'], 0)
            self.assertEqual(event['usage']['input_tokens'], 17)
            self.assertEqual(event['usage']['output_tokens'], 11)
        self.assertEqual(events[-1]['request']['max_output_tokens'], 64000)
        summary_payload = json.loads(events[-1]['request']['input'][1]['content'])
        self.assertIn('reference_aliases', summary_payload)
        wire = Path(result['wire']).read_text()
        self.assertNotIn('PRIVATE-COT', wire)
        self.assertNotIn('synthetic-key-never-persist', wire)
        self.assertTrue(events[-1]['record_changed'])
        phases = [event['phase'] for event, _ in hub.events]
        self.assertIn('maintenance_propose_started', phases)
        self.assertIn('maintenance_publish_result', phases)
        saved = json.loads(Path(result['batch_path']).read_text())
        self.assertEqual(saved['messages'][0]['id'], 'chat#1')
        self.assertEqual(saved['messages'][0]['text'], '我喜欢合成测试中的蓝色。')
        # A second chat can use this Published; injection is the caller's switch.
        self.assertTrue(summary.published().present)
        reloaded = self.helper(config)
        reloaded.initialize()
        self.assertEqual(reloaded.published().text, summary.published().text)
        self.assertEqual(reloaded.status()['usage']['model_calls'], 3)
        disabled = SandboxSummary(config, options={'enabled': False})
        with patch('continuity.sandbox_summary.subprocess.run', side_effect=AssertionError('engine called')):
            self.assertEqual(disabled.published().text, summary.published().text)
            self.assertEqual(disabled.update(conn, *ids)['status'], 'disabled')

    def test_wire_relay_arrives_while_runner_blocked_and_drains_once_on_exit(self):
        for mode in ('ok', 'exception', 'timeout', 'recorder_error'):
            with self.subTest(mode=mode):
                config = self.config(mode)
                conn = self.connection(config)
                summary = self.helper(config)
                received = threading.Event()
                events, callback_threads = [], []

                def recorder(event):
                    events.append(event)
                    callback_threads.append(threading.current_thread())
                    received.set()
                    if mode == 'recorder_error' and event['call'] == 1:
                        raise RuntimeError('synthetic private error body must not be recorded')

                def runner(batch, work_dir):
                    path = work_dir / 'fixture.wire.jsonl'
                    first = {'type': 'provider_call', 'role': 'summary', 'call': 1}
                    path.write_text(json.dumps(first) + '\n', encoding='utf-8')
                    # The later engine stage cannot finish until the first
                    # completed HTTP call has already reached the recorder.
                    self.assertTrue(received.wait(2), 'wire was withheld until runner completion')
                    self.assertEqual(events, [first])
                    with path.open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps({**first, 'call': 2}) + '\n')
                    # No wait here: stop/join must drain the final complete row.
                    if mode == 'exception':
                        raise RuntimeError('synthetic engine failure')
                    if mode == 'timeout':
                        raise subprocess.TimeoutExpired('synthetic-cli', 1)
                    return {'status': 'noop', 'version': 0}

                with patch('continuity.sandbox_summary.CliMaintenanceRunner.run', side_effect=runner):
                    result = summary.update(conn, *self.turn(conn), recorder=recorder)
                self.assertEqual(result['status'], 'failed' if mode in ('exception', 'timeout') else 'noop')
                self.assertEqual([event['call'] for event in events], [1, 2])
                self.assertTrue(all(not thread.is_alive() for thread in callback_threads))
                self.assertEqual(len(Path(result['wire']).read_text().splitlines()), 2)
                self.assertEqual(result['usage']['model_calls'], 0)
                if mode == 'recorder_error':
                    self.assertEqual(result['errors'][0]['type'], 'relay_error')
                    self.assertNotIn('private error body', json.dumps(result))

    def test_custom_and_no_memory_start_clean_but_keep_fixed_reference(self):
        source, current = self.source()
        for label, initial in [('custom', Published(True, 0, 'Fixed custom synthetic context.')),
                               ('old', replace(current, version=65)), ('none', Published(False))]:
            with self.subTest(label=label):
                config = self.config(label)
                conn = self.connection(config)
                summary = self.helper(config, source_config=source, initial=initial)
                self.assertEqual(summary.initialize()['seed_mode'], 'fixed_context_plus_growing_session_summary')
                result = summary.update(conn, *self.turn(conn))
                self.assertEqual(result['status'], 'ok', result)
                text = summary.published().text
                self.assertIn('我喜欢合成测试中的蓝色', text)
                if initial.text:
                    self.assertIn(initial.text, text)
                    self.assertIn('not imported as structured claims', text)
                with closing(sqlite3.connect(config.maint_state_dir / 'state.sqlite3')) as engine:
                    rows = engine.execute('SELECT text FROM messages').fetchall()
                    self.assertEqual(len(rows), 2)
                    self.assertNotIn('Synthetic baseline evidence.', str(rows))

    def test_failure_no_transport_retry_preserves_previous_memory_and_answer(self):
        config = self.config()
        conn = self.connection(config)
        summary = self.helper(config)
        good = summary.update(conn, *self.turn(conn))
        before = summary.published()
        self.cli.with_suffix('.fail').touch()
        result = summary.update(conn, *self.turn(conn, '我喜欢另一个合成测试颜色。'))
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(summary.published(), before)
        self.assertEqual(result['before_text'], result['after_text'])
        self.assertEqual(result['usage']['delta']['requests'], good['usage']['delta']['requests'] + 1)
        self.assertEqual(conn.execute('SELECT content FROM chat_log ORDER BY id DESC').fetchone()[0],
                         'Synthetic answer retained.')
        self.assertEqual(len(Path(result['wire']).read_text().splitlines()), 1)
        for base in (config.maint_state_dir, Path(result['work_dir'])):
            for path in base.rglob('*'):
                if path.suffix in ('.json', '.jsonl'):
                    self.assertNotIn('PRIVATE-COT', path.read_text(), str(path))
                    self.assertNotIn('synthetic-key-never-persist', path.read_text(), str(path))

    def test_engine_bounded_repair_requests_are_all_captured(self):
        self.cli.with_suffix('.repair').touch()
        config = self.config()
        conn = self.connection(config)
        events = []
        summary = self.helper(config)
        result = summary.update(conn, *self.turn(conn), recorder=events.append)
        self.assertEqual(result['status'], 'ok', result)
        self.assertEqual(len(events), 4)
        self.assertEqual(result['usage']['model_calls'], 4)
        payloads = [json.loads(e['request']['input'][1]['content']) for e in events]
        self.assertIn('previous_attempt', payloads[1])
        self.assertEqual(payloads[0]['new_messages'], payloads[1]['new_messages'])

    def test_explicit_retry_keeps_exact_batch_and_duplicate_is_idempotent(self):
        config = self.config()
        conn = self.connection(config)
        ids = self.turn(conn)
        summary = self.helper(config)
        self.cli.with_suffix('.fail').touch()
        failed = summary.update(conn, *ids)
        self.assertEqual(failed['status'], 'failed')
        original = Path(failed['batch_path']).read_bytes()
        self.cli.with_suffix('.fail').unlink()
        retried = summary.update(conn, *ids)
        self.assertEqual(retried['status'], 'ok', retried)
        self.assertEqual(original, Path(retried['batch_path']).read_bytes())
        duplicate = summary.update(conn, *ids)
        self.assertEqual(duplicate['status'], 'duplicate', duplicate)
        self.assertEqual(original, Path(duplicate['batch_path']).read_bytes())
        self.assertEqual(duplicate['usage']['model_calls'], 4)

    def test_baseline_cap_and_case_isolation_with_original_source_chat_ids(self):
        source, initial = self.source()
        before = hashes(source.maint_state_dir)
        configs = [self.config('case1'), self.config('case2')]
        for config in configs:
            conn = self.connection(config)
            # Source chat IDs retained by the copied chat DB; new turns begin #3.
            self.turn(conn, 'Source chat rows that must not be re-ingested.')
            summary = SandboxSummary(config, source_config=source, initial=initial,
                                     options={'enabled': True, 'max_requests': 3})
            result = summary.update(conn, *self.turn(conn))
            self.assertEqual(result['status'], 'ok', result)
            self.assertEqual(result['usage']['baseline']['requests'], 298)
            self.assertEqual(result['usage']['total']['requests'], 301)
            self.assertEqual(result['usage']['model_calls'], 3)
            blocked = summary.update(conn, *self.turn(conn, '我喜欢预算测试。'))
            self.assertEqual(blocked['status'], 'failed')
            self.assertEqual(blocked['errors'][0]['type'], 'budget_exhausted')
            self.assertEqual(blocked['usage']['total']['requests'], 301)
            with closing(sqlite3.connect(config.maint_state_dir / 'state.sqlite3')) as engine:
                self.assertEqual(engine.execute('SELECT COUNT(*) FROM messages').fetchone()[0], 3)
        self.assertEqual(before, hashes(source.maint_state_dir))
        self.assertNotEqual(read_published(configs[0].maint_state_dir, source.user_id).summary_path,
                            read_published(configs[1].maint_state_dir, source.user_id).summary_path)

    def test_snapshot_race_fails_closed_before_publisher_or_model(self):
        source, initial = self.source()
        config = self.config()
        original = sqlite3.connect

        def connect(path, *args, **kwargs):
            conn = original(path, *args, **kwargs)
            if str(path).startswith('file:'):
                with closing(original(source.maint_state_dir / 'state.sqlite3')) as writer:
                    writer.execute('UPDATE users SET version=version+1')
                    writer.commit()
            return conn

        with patch('continuity.sandbox_summary.sqlite3.connect', side_effect=connect), \
             patch('continuity.sandbox_summary.subprocess.run', side_effect=AssertionError('publisher called')):
            with self.assertRaisesRegex(ValueError, 'advanced'):
                self.helper(config, source_config=source, initial=initial).initialize()

    def test_settings_and_path_guards(self):
        config = self.config()
        for options in [{'enabled': True}, {'enabled': True, 'max_requests': 1001},
                        {'enabled': True, 'max_requests': 1, 'model': 'solhigh'},
                        {'max_output_tokens': 16000}, {'command': 'arbitrary'}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                SandboxSummary(config, options=options)
        source, initial = self.source()
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.helper(source, source_config=source, initial=initial).initialize()



if __name__ == '__main__':
    unittest.main()
