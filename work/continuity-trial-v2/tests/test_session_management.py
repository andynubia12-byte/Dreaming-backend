"""Small session-management checks using temporary source data and loopback HTTP."""
from contextlib import closing
import hashlib
import http.client
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from continuity import db, workbench
from continuity.config import Config
from continuity.session_management import delete_session, rename_session


class SessionManagementTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        formal = root / 'formal'
        self.source = Config(project_root=formal, db_path=formal / 'chat.sqlite',
            runs_dir=formal / 'runs', usage_file=formal / 'usage.json',
            maint_state_dir=formal / 'state', export_dir=None,
            retrieval_source_db=None, retrieval_manifest=None)
        self.source.ensure_dirs()
        with closing(db.connect(self.source.db_path)) as conn:
            db.init_db(conn)
            conn.execute("INSERT INTO messages(conversation_id,node_id,message_id,role,text,"
                "create_time,is_selected_branch,is_assistant_reasoning) "
                "VALUES('fixture','past','past','user','fixture history',1,1,0)")
            conn.commit()
            conn.execute('PRAGMA journal_mode=DELETE')
        self.baseline = self.hashes(formal)
        self.wb = workbench.Workbench(self.source, root / 'managed')
        for name in ('create_client', 'Workbench.engine'):
            mock = patch('continuity.workbench.' + name,
                         side_effect=AssertionError('no provider or summary calls'))
            mock.start()
            self.addCleanup(mock.stop)

    @staticmethod
    def hashes(root):
        return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob('*') if p.is_file()}

    def tearDown(self):
        self.assertEqual(self.baseline, self.hashes(self.source.project_root))

    def create(self, previous=None):
        body = ({'continue_from': previous} if previous else {
            'model': self.wb.pool.profiles()[0],
            'memory': {'mode': 'context', 'text': 'fixture memory'},
            'summary': {'enabled': False}})
        return self.wb.create_session(body)['session']['id']

    def test_rename_persists_and_recovers(self):
        sid = self.create()
        result = rename_session(self.wb, sid, {'title': '  新标题  '})
        self.assertEqual(result['session']['title'], '新标题')
        self.assertNotIn('config', result['session'])
        recovered = workbench.Workbench(self.source, self.wb.root)
        self.assertEqual(recovered.detail(sid)['session']['title'], '新标题')
        self.assertEqual(recovered.bootstrap()['sessions'][0]['title'], '新标题')

    def test_delete_owner_retains_shared_memory_history_and_results(self):
        owner = self.create()
        child = self.create(owner)
        sibling = self.create(child)
        config = workbench.restore_config(self.wb.sessions[child]['config'])
        with closing(db.connect(config.db_path)) as conn:
            conn.execute('INSERT INTO chat_log(created_at,instance,conversation_id,role,content) '
                'VALUES(1,?,?,?,?)', (config.instance, child, 'user', 'fixture raw chat'))
            conn.commit()
        artifacts = self.wb.root / 'runs' / 'fixture-run'
        record = {'id': 'fixture-run', 'session_id': owner, 'request_id': 'fixture-request',
                  'status': 'completed', 'created_at': 1}
        workbench.write(artifacts / 'run.json', record)
        workbench.write(artifacts / 'answer.json', {'answer': 'saved answer'})
        workbench.write(artifacts / 'summary.json', {'status': 'published'})
        workbench.write(artifacts / 'wire.jsonl', {'fixture': 'saved wire'})
        before = self.hashes(self.wb.root)
        self.assertEqual(delete_session(self.wb, owner),
                         {'ok': True, 'id': owner, 'deleted': True})
        self.assertEqual(delete_session(self.wb, owner)['deleted'], True)
        renamed = rename_session(self.wb, child, {'title': '续会话'})
        self.assertEqual(renamed['session']['id'], child)
        self.assertEqual(workbench.read(self.wb.root / 'sessions' / owner / 'session.json')['title'],
                         '新会话')
        self.assertTrue(workbench.read(self.wb.root / 'sessions' / owner / 'session.json')['deleted_at'])
        self.assertEqual(self.wb.detail(child)['memory']['text'], 'fixture memory')
        self.assertEqual(self.wb.detail(child)['messages'][0]['content'], 'fixture raw chat')
        with closing(db.connect(config.db_path)) as conn:
            self.assertEqual(conn.execute('SELECT text FROM messages').fetchone()[0], 'fixture history')
        delete_session(self.wb, child)
        after = self.hashes(self.wb.root)
        changed = {f'sessions/{sid}/session.json' for sid in (owner, child)}
        self.assertEqual({k: v for k, v in before.items() if k not in changed},
                         {k: v for k, v in after.items() if k not in changed})
        recovered = workbench.Workbench(self.source, self.wb.root)
        self.assertEqual(set(recovered.sessions), {sibling})
        self.assertEqual({s['id'] for s in recovered.bootstrap()['sessions']}, {sibling})
        self.assertEqual(recovered.detail(sibling)['memory']['text'], 'fixture memory')
        self.assertIn('fixture-run', recovered.runs)
        self.assertTrue(recovered.locks[sibling] is recovered.locks[owner])
        with self.assertRaises(workbench.HTTPError) as error:
            recovered.detail(owner)
        self.assertEqual(error.exception.status, 404)

    def test_busy_shared_workspace_rejects_changes_without_writes(self):
        owner = self.create()
        child = self.create(owner)
        before = self.hashes(self.wb.root)
        lock = self.wb.locks[owner]
        lock.acquire()
        try:
            with self.assertRaises(workbench.HTTPError) as error:
                delete_session(self.wb, child)
            self.assertEqual(error.exception.status, 409)
        finally:
            lock.release()
        # A registered request must also block mutations before its worker starts.
        self.wb.runs['accepted-fixture'] = {'session_id': owner, 'status': 'accepted'}
        with self.assertRaises(workbench.HTTPError) as error:
            rename_session(self.wb, child, {'title': 'busy'})
        self.assertEqual(error.exception.status, 409)
        self.assertEqual(before, self.hashes(self.wb.root))

    def test_loopback_rename_delete_routes(self):
        sid = self.create()
        server = ThreadingHTTPServer(('127.0.0.1', 0), workbench.make_handler(self.wb))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            def request(method, path, body=None):
                with closing(http.client.HTTPConnection('127.0.0.1', server.server_port,
                                                        timeout=5)) as connection:
                    connection.request(method, '/workbench/' + path,
                        body=json.dumps(body) if body is not None else None,
                        headers={'Content-Type': 'application/json'})
                    response = connection.getresponse()
                    return response.status, json.loads(response.read())

            status, result = request('POST', f'sessions/{sid}/rename', {'title': 'HTTP title'})
            self.assertEqual(status, 200, result)
            self.assertEqual(result['session']['title'], 'HTTP title')
            status, result = request('DELETE', f'sessions/{sid}')
            self.assertEqual(status, 200, result)
            self.assertEqual(result, {'ok': True, 'id': sid, 'deleted': True})
            self.assertEqual(request('GET', f'sessions/{sid}')[0], 404)
            self.assertEqual(request('GET', 'bootstrap')[1]['sessions'], [])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)


if __name__ == '__main__':
    unittest.main()
