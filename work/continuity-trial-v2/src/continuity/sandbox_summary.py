"""Optional real maintenance on a caller-owned, isolated workbench branch.

An exact current Published can seed a full structured SQLite snapshot. Every
other selection stays fixed reference context, beside a new engine branch whose
only evidence is subsequent test turns. Callers serialize update with chat and
independently choose whether published() is injected into chat or only previewed.
No workers are started here.
"""
from __future__ import annotations

from contextlib import closing
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import threading
import uuid

from .config import PROJECT_ROOT, Config
from .maintenance import CliMaintenanceRunner, OK_STATUSES, build_turn_batch
from .providers import Profile, redact_record
from .summary_store import Published, read_published

_WRAPPER = Path(__file__).with_name('sandbox_summary_cli.py')
MAX_NEW_REQUESTS = 1000


def _write(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), 'utf-8')
    temporary.replace(path)


def _identity(published):
    return (published.present, published.user_id, published.version,
            hashlib.sha256(published.text.encode('utf-8')).hexdigest())


def _row_id(value):
    """Positive SQLite chat_log IDs, as integers, decimal strings or chat#N."""
    if isinstance(value, bool) or not re.fullmatch(r'(?:chat#)?[1-9][0-9]*', str(value)):
        raise ValueError('summary message IDs must be positive integers or chat#N')
    return int(str(value).removeprefix('chat#'))


def _ledger(state):
    path = Path(state) / 'ledger' / 'usage.json'
    raw = json.loads(path.read_text('utf-8')) if path.exists() else {}
    result = {key: raw.get(key, 0) for key in ('requests', 'input_tokens', 'output_tokens', 'errors')}
    if any(type(value) is not int or value < 0 for value in result.values()):
        raise ValueError('invalid summary ledger; refusing to reset cumulative usage')
    return result


class SandboxSummary:
    def __init__(self, config: Config, *, source_config: Config | None = None,
                 initial: Published | None = None, options: dict | None = None):
        self.config = replace(config)
        self.source_config = source_config
        self.initial = replace(initial) if initial is not None else Published(False, user_id=config.user_id)
        options = dict(options or {})
        if set(options) - {'enabled', 'model', 'profile', 'max_requests', 'max_output_tokens'}:
            raise ValueError('unknown summary option')
        self.enabled = options.get('enabled', False)
        self.model = options.get('model', config.maint_model)
        self.new_budget = options.get('max_requests', config.maint_total_budget)
        output = options.get('max_output_tokens', 64000)
        if not isinstance(self.enabled, bool):
            raise ValueError('summary enabled must be boolean')
        if self.enabled and 'max_requests' not in options:
            raise ValueError('enabled summary maintenance requires an explicit additional request budget')
        self._runtime_selected = options.get('profile') is not None or 'model' in options
        self.profile = copy.deepcopy(options.get('profile'))
        if self.profile is None:
            if not isinstance(self.model, str) or not re.fullmatch(r'deepseek-[a-z0-9][a-z0-9.-]{0,63}', self.model):
                raise ValueError('summary model convenience requires a DeepSeek model; otherwise provide a Responses profile')
            self.profile = {'id': 'summary', 'model': self.model,
                            'base_url': config.api_base_url, 'api_format': 'responses',
                            'api_key_env': config.api_key_env,
                            'parameters': {'reasoning': {'effort': 'high'}}}
        if not isinstance(self.profile, dict) or self.profile.get('api_format') != 'responses':
            raise ValueError('summary profile requires Responses API')
        self.model = Profile(self.profile).model
        if (type(self.new_budget) is not int or not 0 <= self.new_budget <= MAX_NEW_REQUESTS
                or type(output) is not int or output != 64000):
            raise ValueError('summary requires 0..1000 additional requests and 64000 output tokens')
        self.config.maint_model = self.model
        self.config.maint_max_output_tokens = output
        self._metadata = None
        self._generated = Published(False, user_id=config.user_id)
        self._last_status = 'uninitialized'

    @property
    def _metadata_path(self):
        return Path(self.config.maint_state_dir) / 'sandbox-summary.json'

    def _validate_paths(self):
        paths = [Path(getattr(self.config, name)).absolute() for name in
                 ('db_path', 'runs_dir', 'usage_file', 'maint_state_dir')]
        for path in paths:
            if '..' in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
                raise ValueError('summary write paths cannot contain symlinks or traversal')
        # Never accept the ordinary formal state, even without a source Config.
        # Tests may use a temporary project_root with all write paths inside it.
        root = Path(self.config.project_root).resolve()
        if not all(path.is_relative_to(root) for path in paths):
            raise ValueError('summary write paths must be inside the isolated project root')
        if root == PROJECT_ROOT and not all(path.is_relative_to(root / 'evaluations') for path in paths):
            raise ValueError('summary write paths must be under evaluations, never formal state')
        state = paths[-1]
        if paths[0].is_relative_to(state) or paths[2].is_relative_to(state):
            raise ValueError('chat database and usage ledger must be separate from engine state')
        if self.source_config:
            sources = [Path(getattr(self.source_config, name)).resolve() for name in
                       ('db_path', 'runs_dir', 'usage_file', 'maint_state_dir')]
            if any(a.is_relative_to(b) or b.is_relative_to(a) for a in paths for b in sources):
                raise ValueError('sandbox writes overlap source paths')

    def initialize(self) -> dict:
        """No model construction/calls; source Store is NEVER opened writable."""
        if self._metadata is not None:
            return self.status()
        self._validate_paths()
        state = Path(self.config.maint_state_dir)
        if self._metadata_path.exists():
            metadata = json.loads(self._metadata_path.read_text('utf-8'))
            if (metadata['user_id'] != self.config.user_id
                    or metadata['initial_identity'] != list(_identity(self.initial))
                    or self.enabled and metadata['new_budget'] != self.new_budget
                    or metadata['engine_cli'] != str(Path(self.config.maint_cli).resolve())):
                raise ValueError('existing summary branch has different seed, user, engine or budget')
            if not self._runtime_selected and metadata.get('profile') is not None:
                from .providers import migrate_legacy_profile
                self.profile = migrate_legacy_profile(metadata['profile'])
                self.model = Profile(self.profile).model
                self.config.maint_model = self.model
            # Runtime transport metadata can advance without resetting the seed,
            # accepted publication, baseline or cumulative engine ledger.
            if metadata.get('profile') != self.profile or metadata.get('model') != self.model:
                metadata.update(profile=copy.deepcopy(self.profile), model=self.model)
                _write(self._metadata_path, metadata)
            self._metadata = metadata
            self.new_budget = metadata['new_budget']
            self._generated = Published(**metadata['accepted'])
            self.config.maint_total_budget = metadata['baseline']['requests'] + self.new_budget
            self._last_status = metadata.get('last_status', 'ready') if self.enabled else 'disabled'
            return self.status()
        if not self.enabled:
            self._last_status = 'disabled'
            return self.status()
        if state.exists() and any(state.iterdir()):
            raise ValueError('summary initialization requires an empty state directory')
        state.mkdir(parents=True, exist_ok=True)
        seed_mode = 'fixed_context_plus_growing_session_summary'
        baseline = dict.fromkeys(('requests', 'input_tokens', 'output_tokens', 'errors'), 0)
        source = self.source_config
        selected = self.initial
        if source and selected.present and selected.text and selected.version > 0:
            current = read_published(source.maint_state_dir, source.user_id)
            if _identity(current) == _identity(selected) and self.config.user_id == source.user_id:
                source_db = Path(source.maint_state_dir) / 'state.sqlite3'
                # SQLite backup includes committed WAL and is read-only at source.
                # A closed snapshot has no WAL: immutable avoids creating source
                # sidecars. A live WAL database requires normal read-only backup.
                query = '?mode=ro' if Path(str(source_db) + '-wal').exists() else '?mode=ro&immutable=1'
                with closing(sqlite3.connect(source_db.resolve().as_uri() + query, uri=True)) as src:
                    with closing(sqlite3.connect(state / 'state.sqlite3')) as dst:
                        src.backup(dst)
                        row = dst.execute('SELECT version FROM users WHERE user_id=?',
                                          (source.user_id,)).fetchone()
                        texts = dst.execute('SELECT summary_text FROM versions WHERE user_id=? AND version=?',
                                            (source.user_id, selected.version)).fetchone()
                        if not row or row[0] != selected.version or not texts or texts[0] != selected.text:
                            raise ValueError('source structured state advanced or differs from selected summary')
                if _identity(read_published(source.maint_state_dir, source.user_id)) != _identity(selected):
                    raise ValueError('source publication advanced during summary snapshot')
                baseline = _ledger(source.maint_state_dir)
                seed_mode = 'cloned_current_structured_state'
        (state / 'ledger').mkdir(exist_ok=True)
        _write(state / 'ledger' / 'usage.json', baseline)
        # Recreate just the latest immutable summary/material/evidence using the
        # existing publisher and the backed-up complete structured database.
        proc = subprocess.run([sys.executable, str(_WRAPPER), '--state-dir', str(state),
                               '--engine-cli', str(self.config.maint_cli),
                               '--publish-user', self.config.user_id],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              timeout=self.config.maint_timeout_seconds)
        if proc.returncode:
            raise ValueError('isolated engine publisher initialization failed')
        self._generated = read_published(state, self.config.user_id)
        if seed_mode == 'cloned_current_structured_state' and _identity(self._generated) != _identity(selected):
            raise ValueError('isolated publication does not match selected summary SHA256')
        self._metadata = {'schema': 1, 'user_id': self.config.user_id, 'seed_mode': seed_mode,
                          'source_mode': ('current_structured' if seed_mode == 'cloned_current_structured_state'
                                          else 'fixed_context' if selected.text else 'none'),
                          'initial_identity': list(_identity(selected)), 'baseline': baseline,
                          'new_budget': self.new_budget,
                          'profile': copy.deepcopy(self.profile), 'model': self.model,
                          'engine_cli': str(Path(self.config.maint_cli).resolve()),
                          'accepted': asdict(self._generated), 'last_status': 'ready'}
        self.config.maint_total_budget = baseline['requests'] + self.new_budget
        self._last_status = 'ready'
        _write(self._metadata_path, self._metadata)
        return self.status()

    def published(self) -> Published:
        self.initialize()
        if not self._generated.present:
            return replace(self.initial)
        if self._metadata['seed_mode'] == 'cloned_current_structured_state' or not self.initial.text:
            return replace(self._generated)
        return replace(self._generated, text=(
            '# Fixed selected reference (not imported as structured claims)\n\n'
            + self.initial.text + '\n\n# Growing session summary (new test turns only)\n\n'
            + self._generated.text), summary_path=None, memory_path=None,
            reason='fixed_context_plus_growing_session_summary')

    def _usage(self):
        baseline = (self._metadata or {}).get('baseline', dict.fromkeys(
            ('requests', 'input_tokens', 'output_tokens', 'errors'), 0))
        total = _ledger(self.config.maint_state_dir) if self._metadata else dict(baseline)
        if any(total[k] < baseline[k] for k in baseline):
            raise ValueError('summary ledger regressed below copied baseline')
        return {'baseline': dict(baseline),
                'delta': {k: total[k] - baseline[k] for k in baseline}, 'total': total,
                'new_budget': self.new_budget, 'cap': baseline['requests'] + self.new_budget,
                'model_calls': total['requests'] - baseline['requests']}

    def status(self) -> dict:
        return {'enabled': self.enabled, 'status': self._last_status, 'model': self.model,
                'capability': 'responses', 'profile': copy.deepcopy(self.profile),
                'max_output_tokens': self.config.maint_max_output_tokens,
                'seed_mode': (self._metadata or {}).get('seed_mode', 'uninitialized'),
                'source_mode': (self._metadata or {}).get('source_mode', 'fixed_context' if self.initial.text else 'none'),
                'version': self._generated.version if self._generated.present else self.initial.version,
                'usage': self._usage()}

    def update(self, conn, user_message_id: int | str, assistant_message_id: int | str,
               *, recorder=None) -> dict:
        self.initialize()
        before = self.published()
        result = {'status': 'disabled', 'version': before.version, 'before_text': before.text,
                  'after_text': before.text, 'changes': [], 'usage': self._usage(), 'errors': []}
        if not self.enabled:
            return result
        work_dir = Path(self.config.runs_dir) / 'summary' / uuid.uuid4().hex
        work_dir.mkdir(parents=True)
        wire_path = None
        wire_stop = None
        wire_thread = None
        relay_errors = []
        outcome = {}
        try:
            uid, aid = _row_id(user_message_id), _row_id(assistant_message_id)
            batch = build_turn_batch(conn, self.config, uid, aid)
            turn_dir = Path(self.config.runs_dir) / 'summary' / f'turn-chat{uid}-chat{aid}'
            turn_dir.mkdir(exist_ok=True)
            saved_batch = turn_dir / 'batch.json'
            if saved_batch.exists():
                original = json.loads(saved_batch.read_text('utf-8'))
                if original['messages'] != batch['messages'] or original['user_id'] != batch['user_id']:
                    raise ValueError('recorded summary batch differs from chat rows; retry refused')
                batch = original
            else:
                _write(saved_batch, batch)
            # Record the exact raw-ID/time batch once; no queue or auto retry.
            _write(work_dir / 'batch.json', batch)
            roles = [m['role'] for m in batch['messages']]
            if uid == aid or roles != ['user', 'assistant']:
                raise ValueError('summary requires a user row followed by an assistant row')
            if self._usage()['delta']['requests'] >= self.new_budget:
                outcome = {'status': 'budget_exhausted'}
            else:
                runner_config = replace(self.config, maint_cli=_WRAPPER)
                if recorder is not None:
                    wire_stop = threading.Event()

                    def relay():
                        cursor = 0

                        def drain():
                            nonlocal cursor
                            try:
                                path = next(work_dir.glob('*.wire.jsonl'), None)
                                if path is None:
                                    return
                                with path.open('rb') as stream:
                                    stream.seek(cursor)
                                    while line := stream.readline():
                                        if not line.endswith(b'\n'):
                                            break
                                        cursor = stream.tell()
                                        try:
                                            recorder(json.loads(line))
                                        except Exception as exc:
                                            relay_errors.append({'type': 'relay_error',
                                                'error_type': type(exc).__name__,
                                                'message': 'summary wire relay failed; artifact retained'})
                            except Exception as exc:
                                relay_errors.append({'type': 'relay_error',
                                    'error_type': type(exc).__name__,
                                    'message': 'summary wire relay failed; artifact retained'})

                        while not wire_stop.wait(0.02):
                            drain()
                        # The runner has exited (or killed/waited its child).
                        # Drain complete trailing records once before joining.
                        drain()

                    wire_thread = threading.Thread(target=relay, name='sandbox-summary-wire')
                    wire_thread.start()
                outcome = CliMaintenanceRunner(runner_config).run(batch, work_dir)
            if outcome.get('status') in OK_STATUSES:
                generated = read_published(self.config.maint_state_dir, self.config.user_id)
                if outcome.get('version', 0) > 0 and not generated.present:
                    raise ValueError('engine result has no valid published summary')
                self._generated = generated
                result['status'] = outcome['status']
            else:
                result['status'] = 'failed'
                result['errors'] = [{'type': str(outcome.get('status', 'engine_failure')),
                                     'message': 'summary maintenance failed; previous memory retained'}]
        except Exception as exc:
            result['status'] = 'failed'
            result['errors'] = [{'type': type(exc).__name__,
                                 'message': 'summary maintenance failed; previous memory retained'}]
        finally:
            if wire_thread is not None:
                wire_stop.set()
                wire_thread.join()
            result['errors'].extend(relay_errors)
            # The wire is a separate artifact: the main chat recorder cannot
            # truthfully claim it captured the maintenance subprocess calls.
            wires = list(work_dir.glob('*.wire.jsonl'))
            if wires:
                wire_path = wires[0]
            after = self.published()
            result.update(version=after.version, after_text=after.text, usage=self._usage(),
                          seed_mode=self._metadata['seed_mode'], source_mode=self._metadata['source_mode'],
                          wire=str(wire_path) if wire_path else None,
                          wire_captured=wire_path is not None, work_dir=str(work_dir),
                          batch_path=str(work_dir / 'batch.json'),
                          changes={'changed': before.text != after.text,
                                   'before_version': before.version, 'after_version': after.version,
                                   'claims_applied': outcome.get('claims_applied', 0)})
            self._last_status = result['status']
            self._metadata.update(accepted=asdict(self._generated), last_status=self._last_status)
            _write(self._metadata_path, self._metadata)
            # Only a safe outcome projection; engine error strings can echo keys.
            _write(work_dir / 'outcome.json', redact_record(result))
        return result
