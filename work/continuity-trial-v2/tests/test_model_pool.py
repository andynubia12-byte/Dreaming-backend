"""Offline model-pool storage/credential checks; synthetic environment only."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from continuity.config import PROJECT_ROOT
from continuity.evaluation import _profile
from continuity.model_pool import ModelPool


class MemoryCredentials:
    def __init__(self):
        self.values = {}

    def set(self, ref, key):
        self.values[ref] = key

    def get(self, ref):
        return self.values.get(ref)

    def delete(self, ref):
        self.values.pop(ref, None)


def profile(model_id='provider-a', *, api_format='responses', **updates):
    return {'id': model_id, 'model': 'vendor/shared-model',
            'base_url': 'https://provider-a.invalid/v1', 'api_format': api_format,
            'api_key_env': 'EXISTING_PROVIDER_KEY', 'parameters': {}, **updates}


def subscription():
    return profile('sol-high', model='sol', auth='codex_subscription',
                   base_url='https://chatgpt.com/backend-api/codex',
                   parameters={'reasoning': {'effort': 'high'}, 'service_tier': 'default'}, label='Sol high')


class ModelPoolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.env = {'EXISTING_PROVIDER_KEY': 'synthetic-existing-key',
                    'DEEPSEEK_API_KEY': 'synthetic-deepseek-key'}
        self.credentials = MemoryCredentials()

    def pool(self, defaults=None, *, root=None, environ=None):
        return ModelPool(root or self.root / 'pool', defaults or [],
                         environ=self.env if environ is None else environ,
                         credential_store=self.credentials)

    def test_multiple_providers_responses_same_model_crud_and_reload(self):
        pool = self.pool([subscription()])
        a = pool.save(profile(parameters={'reasoning': {'effort': 'high'},
                                          'text': {'verbosity': 'low'}}), label='Provider A')
        b = pool.save(profile('provider-b', api_format='responses',
                              base_url='https://provider-b.invalid/compat/v1',
                              parameters={'temperature': 0.2, 'max_output_tokens': 4096}),
                      label='Provider B')
        self.assertEqual(a['model'], b['model'])
        self.assertNotEqual(a['id'], b['id'])
        self.assertNotEqual(a['base_url'], b['base_url'])
        self.assertEqual(self.pool([subscription()]).profiles(), pool.profiles())
        self.assertEqual(pool.get(a['id']), a)
        for saved in pool.profiles():
            shared = dict(saved)
            shared.pop('label')
            _profile(shared)
            self.assertEqual(set(saved) - set(shared), {'label'})
        status = {s['id']: s for s in pool.credential_status()}
        self.assertEqual(status[a['id']], {'id': a['id'], 'available': True, 'kind': 'environment'})
        pool.delete(a['id'])
        self.assertEqual(a['label'], 'Provider A')
        with self.assertRaisesRegex(ValueError, 'unknown model ID'):
            pool.get(a['id'])
        self.assertEqual(self.pool([subscription()]).get(b['id']), b)

    def test_new_cc_is_rejected_but_stored_metadata_migrates_without_writes(self):
        legacy_default = profile('builtin', api_format='chat_completions', label='Builtin',
            parameters={'max_tokens': 1000, 'reasoning_effort': 'high'})
        legacy_user = profile('custom', api_format='chat_completions', label='User name',
            base_url='https://custom.invalid/v1/chat/completions',
            parameters={'max_completion_tokens': 2000, 'thinking': {'type': 'disabled'}})
        root = self.root / 'pool'
        root.mkdir()
        file = root / 'models.json'
        file.write_text(json.dumps({'version': 1, 'profiles': [legacy_user]}))
        before = file.read_bytes()
        pool = self.pool([legacy_default])
        for model_id, effort in [('builtin', 'high'), ('custom', 'none')]:
            migrated = pool.get(model_id)
            self.assertEqual(migrated['api_format'], 'responses')
            self.assertEqual(migrated['parameters']['reasoning']['effort'], effort)
            self.assertEqual(migrated['api_key_env'], 'EXISTING_PROVIDER_KEY')
        self.assertEqual(pool.get('custom')['label'], 'User name')
        self.assertEqual(pool.get('custom')['base_url'], 'https://custom.invalid/v1')
        self.assertEqual(pool.get('custom')['parameters']['max_output_tokens'], 2000)
        self.assertEqual(file.read_bytes(), before)
        self.assertEqual({m['id'] for m in pool.protocol_migrations()}, {'builtin', 'custom'})
        with self.assertRaises(ValueError):
            pool.save(legacy_user, api_key='synthetic-rejected-key')
        self.assertEqual(file.read_bytes(), before)
        self.assertNotIn('synthetic-rejected-key', self.env.values())
        for effort in ('none', 'low', 'medium', 'high', 'xhigh', 'max'):
            value = pool.save(profile(parameters={'reasoning': {'effort': effort}}))
            self.assertEqual(value['parameters']['reasoning']['effort'], effort)

    def test_defaults_can_be_overridden_but_not_deleted_and_snapshots_are_detached(self):
        default = profile(label='Builtin')
        pool = self.pool([default])
        old = pool.get(default['id'])
        edited = pool.save(profile(parameters={'reasoning': {'effort': 'high'}}))
        self.assertEqual(edited['id'], old['id'])
        self.assertEqual(edited['label'], 'Builtin')
        self.assertEqual(old['parameters'], {})
        edited['parameters']['reasoning']['effort'] = 'low'
        default['label'] = 'mutated by caller'
        self.assertEqual(pool.get(old['id'])['parameters']['reasoning']['effort'], 'high')
        with self.assertRaisesRegex(ValueError, 'cannot be deleted'):
            pool.delete(old['id'])
        copied = pool.save({**old, 'id': 'copied-default'}, label='Copy')
        pool.delete(copied['id'])
        self.assertEqual(self.pool([profile(label='Builtin')]).get(old['id'])['label'], 'Builtin')

    def test_keychain_keys_restore_after_restart_and_keep_frozen_references(self):
        original_env = dict(self.env)
        pool = self.pool()
        data = profile(label='Display name is not a credential')
        old = pool.save(data, api_key='  synthetic-new-private-key  ')
        first_reference = old['api_key_env']
        self.assertTrue(first_reference.startswith('WORKBENCH_POOL_KEY_'))
        self.assertEqual(self.env[first_reference], 'synthetic-new-private-key')
        rotated = pool.save(old, api_key='synthetic-second-private-key')
        self.assertNotEqual(rotated['api_key_env'], first_reference)
        self.assertEqual(self.env[first_reference], 'synthetic-new-private-key')
        self.assertEqual({k: self.env[k] for k in original_env}, original_env)
        self.assertEqual(data['api_key_env'], 'EXISTING_PROVIDER_KEY')
        self.assertEqual(old['label'], data['label'])
        self.assertEqual(pool.credential_status()[0]['kind'], 'keychain')
        self.assertTrue(pool.credential_status()[0]['available'])
        persisted = (pool.root / 'models.json').read_text()
        for secret in (*original_env.values(), 'synthetic-new-private-key', 'synthetic-second-private-key'):
            self.assertNotIn(secret, persisted)
            self.assertNotIn(secret, repr(rotated))
            self.assertNotIn(secret, repr(pool))
        restarted_env = {}
        restarted = self.pool(environ=restarted_env)
        self.assertEqual(restarted.get(old['id']), rotated)
        self.assertEqual(restarted.credential_status(), [{'id': old['id'], 'available': True, 'kind': 'keychain'}])
        self.assertEqual(restarted_env[first_reference], 'synthetic-new-private-key')
        self.assertEqual(restarted_env[rotated['api_key_env']], 'synthetic-second-private-key')
        self.assertNotIn('DEEPSEEK_API_KEY', self.credentials.values)
        self.assertNotIn('EXISTING_PROVIDER_KEY', self.credentials.values)
        pool.delete(old['id'])
        deleted_restart_env = {}
        self.assertEqual(self.pool(environ=deleted_restart_env).profiles(), [])
        self.assertEqual(deleted_restart_env, restarted_env)
        self.assertEqual(self.env[first_reference], 'synthetic-new-private-key')
        self.assertEqual(self.env[rotated['api_key_env']], 'synthetic-second-private-key')
        self.assertEqual(old['api_key_env'], first_reference)

    def test_optional_key_reference_validation_and_safe_failures(self):
        pool = self.pool()
        data = profile()
        del data['api_key_env']
        saved = pool.save(data, api_key='x')
        self.assertEqual(self.env[saved['api_key_env']], 'x')
        saved = pool.save(data, api_key='z' * 8192)
        self.assertEqual(len(self.env[saved['api_key_env']]), 8192)
        before = dict(self.env)
        persisted = (pool.root / 'models.json').read_bytes()
        for key in ('', '   ', 'synthetic-invalid\nkey', 'synthetic-key\n', 'synthetic\x00key',
                    'synthetic-非ASCII', 'z' * 8193, 123):
            with self.subTest(kind=type(key).__name__), self.assertRaises(ValueError) as caught:
                pool.save(data, api_key=key)
            self.assertNotIn('synthetic', str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            pool.save({**data, 'api_key': 'synthetic-protected-secret'}, api_key='synthetic-valid-key')
        self.assertNotIn('synthetic', repr(caught.exception))
        with self.assertRaises(ValueError):
            pool.save(data, label='This label must never be used as a key')
        self.assertEqual(before, self.env)
        self.assertEqual(persisted, (pool.root / 'models.json').read_bytes())

    def test_subscription_is_unknown_without_auth_reads_or_environment_access(self):
        class NoEnvironment(dict):
            def get(self, *args):
                raise AssertionError('subscription environment was consulted')

            def __contains__(self, key):
                raise AssertionError('subscription environment was consulted')

        with patch('urllib.request.urlopen', side_effect=AssertionError('network used')), \
             patch('continuity.codex_credentials.resolve_codex_headers', side_effect=AssertionError('auth read')), \
             patch('pathlib.Path.home', side_effect=AssertionError('auth discovery')):
            pool = self.pool([subscription()], environ=NoEnvironment())
            pool.save(subscription())
            self.assertEqual(pool.get('sol-high')['parameters']['reasoning']['effort'], 'high')
            self.assertEqual(pool.credential_status(), [
                {'id': 'sol-high', 'available': None, 'kind': 'codex_subscription'}])
            with self.assertRaisesRegex(ValueError, 'subscription profiles'):
                pool.save(subscription(), api_key='synthetic-forbidden-key')
            for invalid in (subscription() | {'base_url': 'https://external.invalid/responses'},
                            subscription() | {'api_format': 'chat_completions'}):
                with self.assertRaises(ValueError):
                    pool.save(invalid)

    def test_shared_profile_validation_blocks_fast_credentials_and_unsafe_urls(self):
        pool = self.pool()
        invalid = [profile(id='../escape'), profile(label='a' * 101), profile(label=5),
                   profile(parameters={'service_tier': 'priority'}),
                   profile(parameters={'service_tier': 'fast'}), profile(parameters={'headers': {}}),
                   profile(parameters={'reasoning': {'effort': 'invalid'}}),
                   profile(parameters={'api_key': 'synthetic-secret'}),
                   profile(api_format='native_vendor_api'), profile(api_key_env='not-a-variable'),
                   profile(base_url='https://user:synthetic-secret@provider.invalid/v1'),
                   profile(base_url='https://provider.invalid/v1?key=synthetic-secret'),
                   profile(base_url='https://provider.invalid/v1#synthetic-secret'),
                   profile(capabilities={'can_do_anything': True})]
        for value in invalid:
            with self.subTest(fields=set(value)), self.assertRaises(ValueError) as caught:
                pool.save(value)
            self.assertNotIn('synthetic-secret', repr(caught.exception))
        self.assertEqual(pool.profiles(), [])
        self.assertEqual(pool.save(profile(), label='名' * 100)['label'], '名' * 100)

    def test_managed_paths_reject_traversal_symlinks_and_formal_root(self):
        for path in (self.root / 'safe' / '..' / 'escape', PROJECT_ROOT, PROJECT_ROOT / 'state',
                     self.root / '.codex' / 'pool', self.root / 'userprofiles'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.pool(root=path)
        target = self.root / 'target'
        target.mkdir()
        link = self.root / 'link'
        link.symlink_to(target, target_is_directory=True)
        for path in (link, link / 'child'):
            with self.assertRaises(ValueError):
                self.pool(root=path)
        pool = self.pool()
        victim = self.root / 'source.json'
        victim.write_text('Synthetic source must not be opened or changed.')
        (pool.root / 'models.json').symlink_to(victim)
        for action in (pool.profiles, pool.credential_status, lambda: pool.get('x'),
                       lambda: pool.save(profile(), api_key='synthetic-never-installed-key'),
                       lambda: pool.delete('custom')):
            with self.assertRaises(ValueError):
                action()
        self.assertEqual(victim.read_text(), 'Synthetic source must not be opened or changed.')
        self.assertNotIn('synthetic-never-installed-key', self.env.values())

    def test_concurrent_instances_merge_without_partial_or_lost_writes(self):
        pools = [self.pool() for _ in range(4)]
        stop = threading.Event()
        reader_errors = []

        def read_atomic():
            while not stop.is_set():
                try:
                    path = pools[0].root / 'models.json'
                    if path.exists():
                        document = json.loads(path.read_text())
                        self.assertEqual(document['version'], 1)
                except Exception as exc:
                    reader_errors.append(type(exc).__name__)
                    return

        reader = threading.Thread(target=read_atomic)
        reader.start()
        try:
            with ThreadPoolExecutor(max_workers=4) as workers:
                results = list(workers.map(lambda i: pools[i % 4].save(profile(f'custom-{i}')), range(16)))
        finally:
            stop.set()
            reader.join()
        self.assertEqual(reader_errors, [])
        self.assertEqual({p['id'] for p in pools[0].profiles()}, {p['id'] for p in results})
        self.assertEqual(list(pools[0].root.glob('.models-*.tmp')), [])

    def test_failed_atomic_save_preserves_profiles_environment_and_cleans_temporary_file(self):
        pool = self.pool()
        old = pool.save(profile())
        before = (pool.root / 'models.json').read_bytes()
        environment = dict(self.env)
        with patch('continuity.model_pool.os.replace', side_effect=OSError('synthetic-private-error')):
            with self.assertRaisesRegex(ValueError, 'could not save') as caught:
                pool.save(profile('new'), api_key='synthetic-not-committed-key')
        self.assertNotIn('synthetic-private-error', repr(caught.exception))
        self.assertEqual((pool.root / 'models.json').read_bytes(), before)
        self.assertEqual(self.env, environment)
        self.assertEqual(self.credentials.values, {})
        self.assertEqual(pool.get(old['id']), old)
        self.assertEqual(list(pool.root.glob('.models-*.tmp')), [])


if __name__ == '__main__':
    unittest.main()
