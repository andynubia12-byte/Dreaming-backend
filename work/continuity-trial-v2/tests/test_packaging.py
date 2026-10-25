"""Small offline packaging checks; synthetic data only, no external requests."""
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from continuity.config import load_config
from continuity.credential_store import CredentialStore
from continuity.workbench import Workbench


class PackagingTests(unittest.TestCase):
    def test_old_path_environment_is_ignored(self):
        with patch.dict(os.environ, {
            'CONTINUITY_MAINT_CLI': '/unavailable-old-project/run_update.py',
            'CONTINUITY_RETRIEVAL_DB': '/unavailable-old-project/history.sqlite',
            'CONTINUITY_EXPORT_DIR': '/unavailable-old-project/export',
            'CONTINUITY_USAGE_FILE': '/unavailable-old-project/usage.json',
        }):
            config = load_config()
        self.assertIsNone(config.export_dir)
        self.assertIsNone(config.retrieval_source_db)
        self.assertIsNone(config.retrieval_manifest)
        for name in ('db_path', 'runs_dir', 'maint_state_dir', 'usage_file', 'maint_cli'):
            self.assertTrue(getattr(config, name).resolve().is_relative_to(ROOT.parent.parent))
        self.assertTrue(config.maint_cli.is_file())

    def test_empty_bootstrap_never_resolves_credentials_or_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = dataclasses.replace(load_config(), project_root=root / 'source',
                db_path=root / 'source/chat.sqlite', runs_dir=root / 'source/runs',
                usage_file=root / 'source/usage.json', maint_state_dir=root / 'source/state')
            with patch.object(CredentialStore, '_libraries', side_effect=AssertionError('Keychain read')), \
                 patch('urllib.request.urlopen', side_effect=AssertionError('provider request')), \
                 patch('continuity.codex_credentials.resolve_codex_headers', side_effect=AssertionError('auth read')):
                result = Workbench(source, root / 'managed').bootstrap()
            self.assertEqual(result['sessions'], [])
            self.assertEqual(result['snapshots'], [])
            self.assertFalse(source.db_path.exists())

    def test_legacy_cli_dry_run_has_complete_local_dependencies(self):
        config = load_config()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch = root / 'batch.json'
            output = root / 'result.json'
            batch.write_text(json.dumps({'user_id': 'synthetic', 'batch_id': 'empty',
                'now': '2026-01-01T00:00:00+08:00', 'messages': []}))
            process = subprocess.run([sys.executable, '-B', str(config.maint_cli),
                '--state-dir', str(root / 'state'), '--batch', str(batch),
                '--output', str(output), '--dry-run'], capture_output=True, text=True, timeout=10)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertTrue(json.loads(output.read_text())['dry_run'])
            self.assertFalse((root / 'state').exists())


if __name__ == '__main__':
    unittest.main()
