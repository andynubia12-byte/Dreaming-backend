"""The controlled factor is model/effort, never memory injection or protocol."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from continuity import chat, db
from continuity.config import Config
from continuity.providers import Profile, create_client, migrate_legacy_profile
from continuity.summary_store import Published
from test_providers import Endpoint, native


class ResponsesInvariantTests(unittest.TestCase):
    def profile(self, model, effort='none'):
        return {'id': model, 'model': model, 'api_format': 'responses',
                'base_url': 'https://fixture.invalid', 'api_key_env': 'INVARIANT_FIXTURE_KEY',
                'parameters': {'reasoning': {'effort': effort}}}

    def test_model_and_effort_never_change_context_roles_or_tools(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {
                'INVARIANT_FIXTURE_KEY': 'local-fixture-only'}):
            root = Path(temporary)
            inputs, tools, bodies = [], [], []
            memory = Published(True, version=66, text='完整常驻记忆\n' + '正文不裁剪。' * 12000)
            for model, effort in [('deepseek-flash', 'none'), ('deepseek-v4-pro', 'high'),
                                  ('external-model', 'medium')]:
                cfg = Config(project_root=root, db_path=root / (model + '.sqlite'),
                    runs_dir=root / model, maint_state_dir=root / model / 'state',
                    usage_file=root / model / 'usage.json', main_chat_model=model)
                conn = db.connect(cfg.db_path)
                try:
                    db.init_db(conn)
                    conn.execute('INSERT INTO chat_log(created_at,instance,conversation_id,role,content) '
                        'VALUES(?,?,?,?,?)', (1799999900, cfg.instance, 'same', 'user', '同一条近期记录'))
                    conn.commit()
                    endpoint = Endpoint(native())
                    with patch('urllib.request.urlopen', endpoint):
                        chat.chat_agentic(cfg, conn, '同一问题\n保留换行。', published=memory,
                            conversation_id='same', now_epoch=1800000000,
                            client=create_client(cfg, self.profile(model, effort)))
                    body = endpoint.bodies[0]
                    self.assertEqual(endpoint.requests[0].full_url, 'https://fixture.invalid/responses')
                    self.assertNotIn('messages', body)
                    self.assertEqual(body['reasoning']['effort'], effort)
                    self.assertEqual([i['role'] for i in body['input']], ['system', 'user'])
                    self.assertIn(memory.text, body['input'][0]['content'])
                    self.assertIn('同一条近期记录', body['input'][0]['content'])
                    inputs.append(body['input']); tools.append(body['tools']); bodies.append(body)
                finally:
                    conn.close()
            self.assertTrue(all(v == inputs[0] for v in inputs))
            self.assertTrue(all(v == tools[0] for v in tools))
            for body in bodies:
                body.pop('model'); body.pop('reasoning')
            self.assertTrue(all(v == bodies[0] for v in bodies))

    def test_legacy_migration_is_explicit_lossless_and_never_weakens_new_validation(self):
        old = self.profile('external-model')
        old.update(api_format='chat_completions', base_url='https://fixture.invalid/v1/chat/completions')
        old['parameters'] = {'max_completion_tokens': 1234, 'reasoning_effort': 'high',
                             'thinking': {'type': 'enabled'}, 'temperature': None}
        frozen = copy.deepcopy(old)
        new = migrate_legacy_profile(old)
        self.assertEqual(old, frozen)
        self.assertEqual(new['parameters'], {'max_output_tokens': 1234,
            'reasoning': {'effort': 'high'}, 'temperature': None})
        self.assertEqual(Profile(new).url, 'https://fixture.invalid/v1/responses')
        with self.assertRaises(ValueError):
            Profile(old)
        old['parameters']['seed'] = 123
        with self.assertRaises(ValueError):
            Profile(migrate_legacy_profile(old))

    def test_no_thinking_survives_saved_json_and_all_agent_roles(self):
        profile = json.loads(json.dumps(self.profile('generic', 'none')))
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {
                'INVARIANT_FIXTURE_KEY': 'local-fixture-only'}):
            cfg = Config(usage_file=Path(temporary) / 'usage.json')
            endpoint = Endpoint(native(), native(), native())
            with patch('urllib.request.urlopen', endpoint):
                for role in ('main', 'retrieval', 'summary'):
                    create_client(cfg, profile, role=role).chat([{'role': 'system', 'content': 'same'},
                        {'role': 'user', 'content': 'same question'}], response_json=role != 'main')
            self.assertEqual(len(endpoint.requests), 3)
            for body, request in zip(endpoint.bodies, endpoint.requests):
                self.assertEqual(body['reasoning'], {'effort': 'none'})
                self.assertTrue(request.full_url.endswith('/responses'))
            self.assertEqual(endpoint.bodies[0]['input'], endpoint.bodies[2]['input'])


if __name__ == '__main__':
    unittest.main()
