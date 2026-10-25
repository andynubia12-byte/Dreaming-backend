"""Core offline checks for the 8788 writer; all evidence is synthetic."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from continuity import db
from continuity.config import Config
from continuity.memory_v1 import MemoryV1, MemoryV1Error, MemoryV1Conflict
from continuity.memory_v1_prompts import EVENT_SYSTEM, STABLE_SYSTEM, STABLE_FINAL, TOOLS
from continuity.providers import create_client, normalize_response

NOW = '2026-10-25T12:00:00+08:00'
EARLIER = '2026-10-24T12:00:00+08:00'


def native(text='', *, calls=(), incomplete=False, reasoning=False):
    output = []
    if reasoning:
        output.append({'type': 'reasoning', 'id': 'reason1',
                       'encrypted_content': 'opaque-fixture',
                       'summary': [], 'content': [{'type': 'reasoning_text', 'text': 'PRIVATE-REASONING'}]})
    for index, (name, args) in enumerate(calls):
        output.append({'type': 'function_call', 'id': 'fc' + str(index),
                       'call_id': 'call' + str(index), 'name': name,
                       'arguments': json.dumps(args), 'status': 'completed'})
    if text:
        output.append({'type': 'message', 'role': 'assistant', 'status': 'completed',
                       'content': [{'type': 'output_text', 'text': text}]})
    return {'id': 'response', 'status': 'incomplete' if incomplete else 'completed',
            'incomplete_details': {'reason': 'max_output_tokens'} if incomplete else None,
            'output': output, 'usage': {'input_tokens': 1, 'output_tokens': 1}}


def event(sid='chat#1', *, date='2026-10-25', text='用户明确喜欢散步。'):
    return native(json.dumps(dict(section_markdown=f'### 更新 {date}\n{text} ({sid})',
                                  summary='新记录', sources=[sid], supersedes=[], confidence=0.9)))


def stable(sid='chat#1', extra=''):
    return native('## A. 稳定背景\n- 用户喜欢散步。 (' + sid + ')\n' + extra +
                  '## B. 交流偏好\n## C. 有时间依据的当前状态\n'
                  '## D. 未找到 / 不确定\n- 未找到明确交流偏好。\n')


class ScriptedClient:
    def __init__(self, *results, hook=None):
        self.results = list(results)
        self.requests = []
        self.hook = hook

    def chat(self, messages, **options):
        self.requests.append((copy.deepcopy(messages), copy.deepcopy(options)))
        if self.hook:
            hook, self.hook = self.hook, None
            hook()
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return normalize_response(result)


class MemoryV1Tests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.config = Config(project_root=self.root, user_id='synthetic-user', instance='synthetic',
                             db_path=self.root / 'sources.sqlite', usage_file=self.root / 'usage.json')
        self.conn = db.connect(self.config.db_path)
        self.addCleanup(self.conn.close)
        db.init_db(self.conn)
        self.conn.executemany('INSERT INTO chat_log(created_at,instance,conversation_id,role,content) '
                              'VALUES(?,?,?,?,?)', [
            (1792800000, 'synthetic', 'one', 'user', '我明确喜欢散步。'),
            (1792886400, 'synthetic', 'one', 'user', '我还喜欢阅读。'),
            (1792886400, 'synthetic', 'one', 'assistant', '你喜欢篮球。'),
            (1792886400, 'another-instance', 'foreign', 'user', '别人的事实。'),
            (1900000000, 'synthetic', 'one', 'user', '未来消息。')])
        self.conn.commit()
        self.engine = MemoryV1(self.root / 'state', self.config)

    def execute(self, client, operation='append', **kw):
        options = dict(conn=self.conn, client=client, now=NOW, operation_id='op', expected_version=0)
        if operation == 'append':
            options['source_ids'] = ['chat#1']
        options.update(kw)
        return self.engine.execute(operation, **options)

    def test_initialize_sorted_batches_atomic_publish_and_full_input(self):
        # Include a full long original; there is no evidence or memory slicing.
        long_text = '重要完整细节' * 4000
        self.conn.execute('UPDATE chat_log SET content=? WHERE id=1', (long_text,))
        self.conn.commit()
        client = ScriptedClient(event(date='2026-10-24'), event('chat#2'))
        result = self.execute(client, 'initialize', batches=[
            dict(now=NOW, source_ids=['chat#2']), dict(now=EARLIER, source_ids=['chat#1'])])
        self.assertEqual(result['status'], 'published')
        self.assertEqual(result['version'], 1)
        self.assertEqual(len(self.engine.versions()), 2)
        self.assertEqual(self.engine.version(0)['event_text'], '')
        self.assertFalse(self.engine.version(0)['present'])
        first = client.requests[0][0][1]['content']
        self.assertIn('# 长期记忆\n', first)
        self.assertIn(long_text, first)
        self.assertIn('2026-10-24', first)
        self.assertIn('### 更新 2026-10-24', client.requests[1][0][1]['content'])
        self.assertEqual(result['memory']['sources'], ['chat#1', 'chat#2'])
        self.assertEqual(self.engine.published().text, result['memory']['event_text'])
        with self.assertRaises(MemoryV1Conflict):
            self.execute(ScriptedClient(), 'initialize', operation_id='init-again',
                         expected_version=1, batches=[dict(now=NOW, source_ids=['chat#1'])])

    def test_append_dedup_and_operation_replay(self):
        first_client = ScriptedClient(event())
        first = self.execute(first_client)
        self.assertEqual(self.execute(ScriptedClient()), first)
        with self.assertRaises(MemoryV1Conflict):
            self.execute(ScriptedClient(), source_ids=['chat#2'])
        noop_client = ScriptedClient()
        noop = self.execute(noop_client, operation_id='dup', expected_version=1,
                            source_ids=['chat#1', 'chat#1'])
        self.assertEqual(noop['status'], 'noop')
        self.assertEqual(noop_client.requests, [])
        client = ScriptedClient(event('chat#2'))
        second = self.execute(client, operation_id='second', expected_version=1,
                              source_ids=['chat#1', 'chat#2', 'chat#2'])
        self.assertEqual(second['version'], 2)
        self.assertTrue(second['memory']['event_text'].startswith(first['memory']['event_text'].rstrip()))
        evidence = client.requests[0][0][1]['content'].split('VERIFIED USER EVIDENCE:\n')[1]
        self.assertNotIn('chat#1', evidence)
        self.assertEqual(evidence.count('chat#2'), 1)
        self.assertEqual(self.execute(ScriptedClient()), first)  # historical replay is stable
        self.assertEqual(self.engine.current()['version'], 2)
        self.assertEqual(MemoryV1(self.root / 'state', self.config).current(), second['memory'])

    def test_failed_final_batch_keeps_zero_then_can_retry(self):
        batches = [dict(now=EARLIER, source_ids=['chat#1']), dict(now=NOW, source_ids=['chat#2'])]
        with self.assertRaises(MemoryV1Error):
            self.execute(ScriptedClient(event(date='2026-10-24'), native('partial', incomplete=True)),
                         'initialize', batches=batches)
        self.assertEqual(self.engine.current()['version'], 0)
        self.assertEqual(len(self.engine.versions()), 1)
        result = self.execute(ScriptedClient(event(date='2026-10-24'), event('chat#2')),
                              'initialize', batches=batches)
        self.assertEqual(result['version'], 1)

    def test_failure_preserves_published_and_budget_does_not_trim(self):
        old = self.execute(ScriptedClient(event()))['memory']
        for index, bad in enumerate([native(''), native('{}'), native('{broken'),
                                     native('answer', incomplete=True), RuntimeError('secret')]):
            with self.subTest(index=index), self.assertRaises(MemoryV1Error):
                self.execute(ScriptedClient(bad), expected_version=1,
                             operation_id=f'bad{index}', source_ids=['chat#2'])
            self.assertEqual(self.engine.current(), old)
        self.config.max_input_tokens = 10
        client = ScriptedClient(event('chat#2'))
        with self.assertRaisesRegex(MemoryV1Error, 'not truncated'):
            self.execute(client, operation_id='budget', expected_version=1, source_ids=['chat#2'])
        self.assertFalse(client.requests)
        self.assertEqual(self.engine.current(), old)

    def test_conflict_during_model_call_never_overwrites_winner(self):
        def competing_publish():
            self.execute(ScriptedClient(event('chat#2')), operation_id='winner', source_ids=['chat#2'])
        with self.assertRaises(MemoryV1Conflict):
            self.execute(ScriptedClient(event(), hook=competing_publish))
        self.assertEqual(self.engine.current()['sources'], ['chat#2'])
        self.assertEqual(self.engine.current()['operation_id'], 'winner')
        self.assertEqual(len(self.engine.versions()), 2)

    def test_all_sources_must_be_user_selected_owned_and_strictly_before_now(self):
        self.conn.executemany('INSERT INTO messages(conversation_id,node_id,role,create_time,text,'
                              'is_selected_branch,is_assistant_reasoning) VALUES(?,?,?,?,?,?,?)', [
            ('export', 'selected', 'user', 1792800000, '导入原话。', 1, 0),
            ('export', 'branch', 'user', 1792800000, '另一分支。', 0, 0),
            ('export', 'reason', 'user', 1792800000, '思考。', 1, 1)])
        from datetime import datetime
        self.conn.execute('INSERT INTO chat_log(created_at,instance,role,content) VALUES(?,?,?,?)',
                          (datetime.fromisoformat(NOW).timestamp(), 'synthetic', 'user', '同时刻消息。'))
        self.conn.commit()
        for index, sid in enumerate(['chat#3', 'chat#4', 'chat#5', 'chat#6', 'chat#missing',
                                      'msg:export#branch', 'msg:export#reason', 'selected']):
            client = ScriptedClient(event())
            with self.subTest(sid=sid), self.assertRaises(MemoryV1Error):
                self.execute(client, operation_id=f'bad-source{index}', source_ids=['chat#1', sid])
            self.assertFalse(client.requests)
        with self.assertRaises(MemoryV1Error):
            self.execute(ScriptedClient(), now='2026-10-25T12:00:00')
        self.assertEqual(self.engine.current()['version'], 0)
        result = self.execute(ScriptedClient(event('msg:export#selected')), operation_id='import',
                              source_ids=['msg:export#selected'])
        self.assertEqual(result['memory']['sources'], ['msg:export#selected'])
        with self.assertRaises(MemoryV1Error):
            MemoryV1(self.root / 'state', replace(self.config, user_id='other'))

    def test_explicit_row_user_ownership_and_concurrent_same_operation_replay(self):
        self.conn.execute('ALTER TABLE messages ADD COLUMN user_id TEXT')
        self.conn.execute('ALTER TABLE chat_log ADD COLUMN user_id TEXT')
        self.conn.execute('UPDATE chat_log SET user_id=?', (self.config.user_id,))
        self.conn.execute("UPDATE chat_log SET user_id='another-user' WHERE id=2")
        self.conn.commit()
        client = ScriptedClient(event('chat#2'))
        with self.assertRaises(MemoryV1Error):
            self.execute(client, operation_id='wrong-user', source_ids=['chat#1', 'chat#2'])
        self.assertFalse(client.requests)
        first = []
        def same_operation():
            first.append(self.execute(ScriptedClient(event(text='首次发布。'))))
        result = self.execute(ScriptedClient(event(text='不会重复发布。'), hook=same_operation))
        self.assertEqual(result, first[0])
        self.assertEqual(len(self.engine.versions()), 2)
        self.assertNotIn('不会重复发布', self.engine.current()['text'])

    def test_stable_autonomous_tools_then_separate_synthesis_and_assembly(self):
        old = self.execute(ScriptedClient(event()))['memory']
        events = []
        long_original = '完整检索证据细节。' * 1000
        self.conn.execute('UPDATE chat_log SET content=? WHERE id=1', (long_original,))
        self.conn.commit()
        rich = '- 明确兴趣和具体背景保留完整。 (chat#1)\n' * 300
        client = ScriptedClient(native(calls=[('search_history', {'query': '散步'})], reasoning=True),
                                native(calls=[('read_message', {'source_id': 'chat#1'}),
                                              ('expand', {'source_id': 'chat#1'})]),
                                native('检索完成'), stable(extra=rich))
        result = self.execute(client, 'refresh_stable', operation_id='refresh', expected_version=1,
                              emit=lambda phase, **data: events.append((phase, data)))
        self.assertEqual(result['version'], 2)
        self.assertEqual(result['memory']['event_text'], old['event_text'])
        self.assertIn(rich, result['memory']['stable_text'])
        self.assertEqual(result['memory']['text'], '# 稳定记忆层（stable.md，实验候选）\n\n' +
            result['memory']['stable_text'] + '\n\n# 完整事件摘要（原 idea.md，逐字保留）\n\n' + old['event_text'])
        final_messages, final_options = client.requests[-1]
        self.assertIsNone(final_options['tools'])
        self.assertEqual(final_messages[-1]['content'], STABLE_FINAL)
        self.assertTrue(all(options['max_tokens'] >= 16000 for _, options in client.requests))
        self.assertIn('_response_items', final_messages[2])
        read_output = next(json.loads(m['content']) for m in final_messages
                           if m['role'] == 'tool' and json.loads(m['content']).get('tool') == 'read_message')
        self.assertEqual(read_output['item']['text'], long_original)
        self.assertFalse(read_output['item']['truncated'])
        self.assertNotIn('PRIVATE-REASONING', json.dumps(events))
        self.assertEqual(self.engine.published().text, result['memory']['text'])

    def test_stable_refuses_unseen_assistant_and_invalid_structure(self):
        for index, (tools, final) in enumerate([
            ([('search_history', {'query': '散步'})], stable('chat#2')),
            ([('read_message', {'source_id': 'chat#3'})], stable('chat#3')),
            ([('read_message', {'source_id': 'chat#1'})], native('缺少章节')),
            ([('read_message', {'source_id': 'chat#1'})], native('', incomplete=True))]):
            client = ScriptedClient(native(calls=tools), native('完成'), final)
            with self.subTest(index=index), self.assertRaises(MemoryV1Error):
                self.execute(client, 'refresh_stable', operation_id='badstable' + str(index))
            self.assertEqual(self.engine.current()['version'], 0)

    def test_noop_empty_event_consumes_sources_without_publishing_seed(self):
        empty = native(json.dumps(dict(section_markdown='', summary='无持久信息', sources=[],
                                       supersedes=[], confidence=1)))
        result = self.execute(ScriptedClient(empty))
        self.assertEqual(result['status'], 'noop')
        self.assertFalse(result['memory']['present'])
        client = ScriptedClient()
        self.assertEqual(self.execute(client, operation_id='repeated')['status'], 'noop')
        self.assertFalse(client.requests)

    def test_native_responses_continuation_and_injected_profile_effort(self):
        bodies = []
        responses = iter([native(calls=[('read_message', {'source_id': 'chat#1'})], reasoning=True),
                          native('检索完成'), stable()])

        def urlopen(request, **kwargs):
            import io
            bodies.append(json.loads(request.data))
            response = io.BytesIO(json.dumps(next(responses)).encode())
            response.status = 200
            return response

        profile = dict(id='injected', model='caller-selected-model', api_format='responses',
                       base_url='https://offline.invalid', api_key_env='MEMORY_V1_FIXTURE_KEY',
                       parameters={'reasoning': {'effort': 'none'}, 'max_output_tokens': 32000})
        client = create_client(self.config, profile, role='summary')
        self.conn.row_factory = None  # the public seam accepts ordinary sqlite connections
        with patch.dict(os.environ, {'MEMORY_V1_FIXTURE_KEY': 'synthetic-secret'}), \
                patch('urllib.request.urlopen', urlopen):
            result = self.execute(client, 'refresh_stable')
        self.assertEqual(result['status'], 'published')
        self.assertEqual(len(bodies), 3)
        self.assertIsNone(self.conn.row_factory)
        for body in bodies:
            self.assertEqual(body['model'], 'caller-selected-model')
            self.assertEqual(body['reasoning'], {'effort': 'none'})
            self.assertEqual(body['max_output_tokens'], 32000)
            self.assertNotIn('previous_response_id', body)
        continuation = bodies[1]['input']
        reasoning = next(i for i in continuation if i.get('type') == 'reasoning')
        self.assertEqual(reasoning['encrypted_content'], 'opaque-fixture')
        call = next(i for i in continuation if i.get('type') == 'function_call')
        output = next(i for i in continuation if i.get('type') == 'function_call_output')
        self.assertEqual(call['call_id'], output['call_id'])
        self.assertEqual(json.loads(output['output'])['item']['role'], 'user')
        database = self.engine.path.read_bytes()
        self.assertNotIn(b'PRIVATE-REASONING', database)
        self.assertNotIn(b'synthetic-secret', database)

    def test_generic_prompt_semantics_are_copied_verbatim(self):
        prompts = json.loads((Path(__file__).resolve().parents[3] /
                              'docs/v1/summary-method/prompts.json').read_text())
        self.assertEqual(EVENT_SYSTEM, prompts['agents-CONSOLIDATE_SYSTEM'])
        self.assertEqual(STABLE_SYSTEM, prompts['stable_tools-SYSTEM'])
        self.assertEqual(STABLE_FINAL, prompts['stable_finalize-FINAL'])
        self.assertEqual(TOOLS, prompts['stable_tools.py:TOOLS'])


if __name__ == '__main__':
    unittest.main()
