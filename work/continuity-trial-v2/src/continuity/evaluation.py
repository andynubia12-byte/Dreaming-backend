"""Isolated batch evaluation; delegates all conversation/retrieval to real chat.

Matrix order, bounded pool and blind/full exports adapt llmtest/index.html's
benchBuildRuns (2902), benchRunPool (3502), benchBlindMarkdown (3843),
benchExportKey (3876), benchExportFull (3900). Audited source SHA-256:
607b76f8004a26ef5250d223abf501d79091933eda2f1df4c2979f209fc0a310.
No runtime dependency on either standalone project; no copied prompt/tool loop.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import html
import itertools
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import closing
from pathlib import Path
from typing import Any

from . import chat, db
from .config import PROJECT_ROOT, Config, load_config
from .summary_store import Published, read_published


class EvaluationError(ValueError):
    """Invalid evaluation input; messages never contain provider error bodies."""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(_json(value) + '\n', encoding='utf-8')
    temporary.replace(path)


def _read(path: Path) -> Any:
    return json.loads(path.read_bytes())


def _keys(value: Any, allowed: set[str], required: set[str] = frozenset()) -> None:
    if not isinstance(value, dict) or set(value) - allowed or required - set(value):
        raise EvaluationError('invalid or unknown configuration fields')


def _number(value: Any, *, integer: bool = False, minimum: float = 0) -> None:
    if (isinstance(value, bool) or not isinstance(value, int if integer else (int, float))
            or not math.isfinite(value) or value < minimum):
        raise EvaluationError('invalid numeric configuration')


def _id(value: Any) -> None:
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', value):
        raise EvaluationError('ids must be short ASCII names without paths')


def _no_symlinks(path: Path) -> Path:
    path = path.absolute()
    if '..' in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise EvaluationError('traversal and symlinks are forbidden')
    return path


def _input(path: Path | str) -> Path:
    path = _no_symlinks(Path(path).expanduser())
    # Never discover credentials; even explicitly selected context files cannot
    # point at the usual auth/secret stores. Only the source Config loader is used.
    if any(p.lower().startswith('.env') or p.lower() in {
            '.codex', '.ssh', 'secrets', 'userprofiles', 'auth.json'} for p in path.parts):
        raise EvaluationError('credential stores are not evaluation inputs')
    return path


def _overlap(a: Path, b: Path) -> bool:
    a, b = a.resolve(), b.resolve()
    return a.is_relative_to(b) or b.is_relative_to(a)


def _destination(kind: str, out: str | Path | None, evaluation_root: Path | None,
                 sources: tuple[Path, ...] = ()) -> Path:
    root = _no_symlinks(evaluation_root or PROJECT_ROOT / 'evaluations')
    target = Path(out) if out is not None else Path(f'{kind}-{uuid.uuid4().hex}')
    if not target.is_absolute():
        target = root / target
    target = _no_symlinks(target)
    if target == root or not target.is_relative_to(root) or target.exists():
        raise EvaluationError('output must be a NEW directory inside evaluations')
    if any(_overlap(target, source) for source in sources):
        raise EvaluationError('output overlaps a source input')
    target.mkdir(parents=True, exist_ok=False)
    return target


def _backup(source: Path, target: Path, *, frozen: bool = False) -> None:
    """SQLite backup includes committed WAL rows; never attach a writable source."""
    source = _input(source)
    query = '?mode=ro&immutable=1' if frozen else '?mode=ro'
    with closing(sqlite3.connect(source.as_uri() + query, uri=True)) as src:
        with closing(sqlite3.connect(target)) as dst:
            src.backup(dst)


def _before(value: Any, now: float) -> None:
    try:
        parsed = dt.datetime.fromisoformat(value)
        if parsed.tzinfo is None or parsed.timestamp() > now:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise EvaluationError('snapshot now_max must be timezone-aware and no later than fixed now') from None


def _seed_rows(conn: sqlite3.Connection, selection: dict) -> list[dict]:
    _keys(selection, {'instance', 'conversation_id', 'limit'}, {'instance', 'conversation_id'})
    for name in ('instance', 'conversation_id'):
        if not isinstance(selection[name], str) or not selection[name]:
            raise EvaluationError('recent selection requires a named session and instance')
    limit = selection.get('limit', 40)
    _number(limit, integer=True, minimum=1)
    rows = conn.execute(
        'SELECT id, role, content, created_at FROM chat_log WHERE instance=? '
        'AND conversation_id=? ORDER BY id DESC LIMIT ?',
        (selection['instance'], selection['conversation_id'], limit)).fetchall()
    return [dict(row) for row in reversed(rows)]


def create_snapshot(*, source_config: Config | None = None, out: str | Path | None = None,
                    empty_summary: bool = False, recent: dict | None = None,
                    evaluation_root: Path | None = None) -> Path:
    """Explicit source read operation only. Run never reads live source state."""
    config = source_config if source_config is not None else load_config()
    source_db = _input(config.db_path)
    source_state = _input(config.maint_state_dir)
    if empty_summary:
        published, summary = Published(True, text='', user_id=config.user_id), b''
    else:
        published = read_published(source_state, config.user_id)
        if not published.present:
            raise EvaluationError('selected published summary is missing, empty or invalid; snapshot refused')
        summary = _input(published.summary_path).read_bytes()
        if summary.decode('utf-8') != published.text:
            raise EvaluationError('selected summary bytes changed or differ from validated text')
    target = _destination('snapshot', out, evaluation_root, (source_db, source_state))
    _backup(source_db, target / 'history.sqlite')
    (target / 'summary.txt').write_bytes(summary)
    with closing(sqlite3.connect((target / 'history.sqlite').as_uri() + '?mode=ro&immutable=1', uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        if conn.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise EvaluationError('source database is corrupt')
        seeds = _seed_rows(conn, recent) if recent else []
        counts = {table: conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
                  for table in ('messages', 'conversations', 'chat_log')}
    metadata = dataclasses.asdict(published)
    metadata.pop('text')
    _write(target / 'snapshot.json', {
        'version': 1, 'summary_sha256': _hash(summary),
        'database_sha256': _file_hash(target / 'history.sqlite'),
        'published': metadata, 'intentional_empty_summary': empty_summary,
        'instance': config.instance, 'user_id': config.user_id, 'recent_seed': seeds,
        'recent_selection': recent,
        'provenance': {'database': str(source_db), 'state': str(source_state),
                       'source_project': str(config.project_root), 'rows': counts,
                       'snapshot_at': dt.datetime.now(dt.timezone.utc).isoformat()},
    })
    return target


def _snapshot(path: Path, now: float) -> dict:
    path = _input(path)
    if any((path / ('history.sqlite' + suffix)).exists() for suffix in ('-wal', '-shm', '-journal')):
        raise EvaluationError('selected snapshot must be closed and immutable, without SQLite sidecars')
    metadata = _read(_input(path / 'snapshot.json'))
    if metadata.get('version') != 1:
        raise EvaluationError('unsupported snapshot version')
    for name, field in (('summary.txt', 'summary_sha256'), ('history.sqlite', 'database_sha256')):
        if _file_hash(_input(path / name)) != metadata.get(field):
            raise EvaluationError('snapshot hash mismatch')
    if metadata['published'].get('now_max') is not None:
        _before(metadata['published']['now_max'], now)
    if not metadata['published'].get('present'):
        raise EvaluationError('snapshot has no selected summary')
    if not (path / 'summary.txt').read_bytes().strip() and not metadata.get('intentional_empty_summary'):
        raise EvaluationError('empty snapshot summary was not explicitly requested')
    return metadata


def _parameters(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if re.search(r'key|token$|secret|auth|password|header|credential', key, re.I):
                raise EvaluationError('credentials/headers are forbidden in parameters')
            _parameters(item)
    elif isinstance(value, list):
        for item in value:
            _parameters(item)


def _profile(profile: dict) -> None:
    _keys(profile, {'id', 'model', 'base_url', 'api_format', 'api_key_env', 'parameters',
                    'timeout_seconds', 'token_limit_field', 'auth'},
          {'id', 'model', 'base_url', 'api_format', 'parameters'})
    _id(profile['id'])
    _parameters(profile['parameters'])
    # Reuse provider's tested protocol/parameter schema, without creating a client
    # or consulting runtime subscription auth during plan/preflight.
    from .providers import Profile
    Profile(profile)


def _seeds(seeds: Any, now: float) -> None:
    if not isinstance(seeds, list):
        raise EvaluationError('recent_seed must be a list or snapshot selector')
    previous = -math.inf
    for row in seeds:
        _keys(row, {'id', 'role', 'content', 'created_at'}, {'role', 'content', 'created_at'})
        _number(row['created_at'])
        if (row['role'] not in ('user', 'assistant') or not isinstance(row['content'], str)
                or row['created_at'] >= now or row['created_at'] < previous):
            raise EvaluationError('recent seeds must be chronological and strictly before fixed now')
        previous = row['created_at']


def load_suite(path: Path | str) -> dict:
    """Validate and materialize explicit files/snapshots once, without any API."""
    path = _input(path)
    raw = path.read_bytes()
    suite = json.loads(raw)
    _keys(suite, {'version', 'models', 'questions', 'contexts', 'repeats', 'concurrency',
                  'budgets', 'now_epoch', 'retrieval_profile', 'recent_seed', 'chat'},
          {'version', 'models', 'questions', 'contexts', 'budgets', 'now_epoch'})
    if suite['version'] != 1:
        raise EvaluationError('unsupported suite version')
    _number(suite['now_epoch'])
    try:
        dt.datetime.fromtimestamp(suite['now_epoch'], dt.timezone.utc)
    except (OSError, OverflowError, ValueError):
        raise EvaluationError('fixed now is outside the supported timestamp range') from None
    for key, default in (('repeats', 1), ('concurrency', 1)):
        suite.setdefault(key, default)
        _number(suite[key], integer=True, minimum=1)
    _keys(suite['budgets'], {'max_requests', 'trial_max_requests', 'max_input_tokens'},
          {'max_requests', 'trial_max_requests', 'max_input_tokens'})
    for value in suite['budgets'].values():
        _number(value, integer=True, minimum=1)
    _keys(suite.get('chat', {}), {'recent_limit', 'recent_token_budget'})
    for value in suite.get('chat', {}).values():
        _number(value, integer=True)
    for group in ('models', 'questions', 'contexts'):
        if not isinstance(suite[group], list) or not suite[group]:
            raise EvaluationError('models/questions/contexts must be nonempty lists')
        seen = set()
        for item in suite[group]:
            if not isinstance(item, dict):
                raise EvaluationError('matrix entries must be objects')
            _id(item.get('id'))
            if item['id'] in seen:
                raise EvaluationError('duplicate matrix id')
            seen.add(item['id'])
    for model in suite['models']:
        _profile(model)
    if 'retrieval_profile' in suite:
        _profile(suite['retrieval_profile'])
    for question in suite['questions']:
        _keys(question, {'id', 'text', 'checks'}, {'id', 'text'})
        if not isinstance(question['text'], str) or not question['text'].strip():
            raise EvaluationError('question text must be nonempty')
        _keys(question.get('checks', {}), {'contains', 'not_contains'})
        for values in question.get('checks', {}).values():
            if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values):
                raise EvaluationError('checks must be lists of nonempty literal strings')
    snapshots = {}
    files = {}
    for context in suite['contexts']:
        _keys(context, {'id', 'mode', 'text', 'file', 'snapshot', 'recent_seed', 'inject_memory'}, {'id', 'mode'})
        mode = context['mode']
        if 'inject_memory' in context and type(context['inject_memory']) is not bool:
            raise EvaluationError('inject_memory must be boolean')
        if mode not in ('none', 'context', 'snapshot', 'pipeline'):
            raise EvaluationError('unknown context mode')
        if mode == 'none':
            if set(context) - {'id', 'mode'}:
                raise EvaluationError('none mode forbids memory and history seeds')
            context['text'] = ''
            context['recent_seed'] = []
            continue
        if mode == 'context':
            if ('text' in context) == ('file' in context) or 'snapshot' in context:
                raise EvaluationError('context mode requires exactly one text/file')
            if 'file' in context:
                file = _input(path.parent / context.pop('file'))
                data = file.read_bytes()
                files[str(file)] = data
                context['text'] = data.decode('utf-8')
            if not isinstance(context['text'], str):
                raise EvaluationError('context text must be a string')
        else:
            if 'snapshot' not in context or 'text' in context or 'file' in context:
                raise EvaluationError('snapshot/pipeline requires a selected snapshot directory')
            source = _input(path.parent / context['snapshot'])
            key = str(source)
            if key not in snapshots:
                metadata = _snapshot(source, suite['now_epoch'])
                snapshots[key] = metadata
            context['snapshot'] = key
            context['text'] = (source / 'summary.txt').read_bytes().decode('utf-8')
            if mode == 'pipeline' and 'retrieval_profile' not in suite:
                raise EvaluationError('pipeline requires a fixed retrieval_profile')
        seeds = context.get('recent_seed', suite.get('recent_seed', []))
        if seeds == 'snapshot' or isinstance(seeds, dict):
            if mode not in ('snapshot', 'pipeline'):
                raise EvaluationError('session seed selection requires a snapshot')
            with closing(sqlite3.connect((Path(context['snapshot']) / 'history.sqlite').as_uri() + '?mode=ro&immutable=1', uri=True)) as conn:
                conn.row_factory = sqlite3.Row
                if seeds == 'snapshot':
                    metadata = snapshots[context['snapshot']]
                    archived = metadata['recent_seed']
                    selection = metadata.get('recent_selection')
                    # Older snapshots archived row IDs without the selector. Recover
                    # that single original session, then verify the exact selected rows.
                    if selection is None and archived:
                        row = conn.execute('SELECT instance, conversation_id FROM chat_log WHERE id=?',
                                           (archived[0].get('id'),)).fetchone()
                        if row is None:
                            raise EvaluationError('snapshot recent rows have no source session')
                        selection = {**dict(row), 'limit': len(archived)}
                    seeds = _seed_rows(conn, selection) if selection is not None else []
                    if seeds != archived:
                        raise EvaluationError('snapshot recent rows disagree with their source session')
                else:
                    selection = dict(seeds)
                    seeds = _seed_rows(conn, selection)
                if selection is not None:
                    context['recent_selection'] = selection
        _seeds(seeds, suite['now_epoch'])
        context['recent_seed'] = seeds
    return {'suite': suite, 'raw': raw, 'path': path, 'files': files, 'snapshots': snapshots}


def build_cases(suite: dict) -> list[dict]:
    cases = []
    for model, context, question, repeat in itertools.product(
            suite['models'], suite['contexts'], suite['questions'], range(1, suite['repeats'] + 1)):
        identity = [model['id'], question['id'], context['id'], repeat]
        cases.append({'index': len(cases), 'case_id': _hash(_json(identity).encode())[:24],
                      'identity': identity, 'model': model, 'context': context,
                      'question': question, 'repeat': repeat})
    return cases


def plan_suite(loaded: dict) -> dict:
    suite = loaded['suite']
    profiles = list(suite['models'])
    if any(c['mode'] == 'pipeline' for c in suite['contexts']):
        profiles.append(suite['retrieval_profile'])
    missing = [{'profile': p['id'], 'api_key_env': p['api_key_env'], 'base_url': p['base_url']}
               for p in profiles if p.get('auth', 'api_key') == 'api_key'
               and not os.environ.get(p['api_key_env'])]
    return {'valid': not missing, 'missing_credentials': missing,
            'cases': [{'case_id': c['case_id'], 'identity': c['identity']} for c in build_cases(suite)],
            'concurrency': suite['concurrency'], 'now_epoch': suite['now_epoch'],
            'per_trial_request_ceiling': min(suite['budgets']['max_requests'], suite['budgets']['trial_max_requests']),
            'runtime_auth_unchecked': [p['id'] for p in profiles if p.get('auth') == 'codex_subscription'],
            'api_calls': 0}


def _redact(value: Any, secrets: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        for secret in sorted(secrets, key=len, reverse=True):
            value = value.replace(secret, '[credential redacted]')
        return value
    if isinstance(value, list):
        return [_redact(v, secrets) for v in value]
    if isinstance(value, dict):
        return {k: _redact(v, secrets) for k, v in value.items()}
    return value


def _isolated_config(folder: Path, case: dict, suite: dict, metadata: dict) -> Config:
    ceiling = min(suite['budgets']['max_requests'], suite['budgets']['trial_max_requests'])
    config = Config(
        project_root=folder, db_path=folder / 'chat.sqlite', runs_dir=folder / 'runs',
        maint_state_dir=folder / 'state', usage_file=folder / 'usage.json',
        export_dir=None, retrieval_source_db=None, retrieval_manifest=None,
        maint_cli=folder / 'disabled-maintenance.py',
        instance=case['context'].get('recent_selection', {}).get('instance', metadata.get('instance', 'evaluation')),
        user_id=metadata.get('user_id', 'evaluation'), api_base_url=case['model']['base_url'],
        api_key_env=case['model'].get('api_key_env', 'EVALUATION_UNUSED_KEY'), main_chat_model=case['model']['model'],
        chat_model=suite.get('retrieval_profile', case['model'])['model'],
        max_requests=ceiling, trial_max_requests=ceiling,
        max_input_tokens=suite['budgets']['max_input_tokens'],
    )
    # Config defaults must never introduce a write path or load live overrides.
    for field in dataclasses.fields(config):
        value = getattr(config, field.name)
        if isinstance(value, Path) and not value.is_relative_to(folder):
            raise EvaluationError('isolated config contains an external path')
    return config


def _checks(answer: str, checks: dict) -> list[dict]:
    return [{'kind': kind, 'literal': value,
             'passed': (value in answer) if kind == 'contains' else (value not in answer)}
            for kind, values in checks.items() for value in values]


def _trial(run: Path, case: dict, suite: dict, snapshots: dict, secrets: tuple[str, ...],
           after_trial=None, initial_usage=None) -> dict:
    folder = run / 'trials' / case['case_id']
    folder.mkdir()
    started = time.perf_counter()
    result = {k: case[k] for k in ('index', 'case_id', 'identity', 'repeat')}
    result.update(status='error', answer='', checks=[], tool_trace=[], usage={}, error=None,
                  wire=f'trials/{case["case_id"]}/wire.jsonl')
    context = case['context']
    metadata = snapshots.get(context.get('snapshot'), {})
    config = _isolated_config(folder, case, suite, metadata)
    if initial_usage is not None:
        _write(config.usage_file, initial_usage)
        result['prior_attempt_usage'] = dict(initial_usage)
    result['context_mode'] = context['mode']
    result['context_injection_enabled'] = context.get('inject_memory', context['mode'] != 'none')
    result['context_sha256'] = _hash(context['text'].encode('utf-8'))
    result['snapshot_sha256'] = metadata.get('database_sha256')
    result['parameters'] = case['model']
    result['fixed_now_epoch'] = suite['now_epoch']
    chat_options = dict(suite.get('chat', {}))
    selection = context.get('recent_selection')
    if selection is not None:
        chat_options['recent_limit'] = min(len(context['recent_seed']),
                                          chat_options.get('recent_limit', config.default_recent_messages))
    result['recent'] = {
        'archived_messages': len(context['recent_seed']), 'source_selection': selection,
        'source_ids': [f'chat#{row["id"]}' for row in context['recent_seed']] if selection is not None else [],
        'requested_chat_options': suite.get('chat', {}),
        'effective_recent_limit': chat_options.get('recent_limit', config.default_recent_messages),
        'effective_recent_token_budget': chat_options.get('recent_token_budget', config.default_recent_token_budget),
    }
    _write(folder / 'config.json', {f.name: str(v) if isinstance(v := getattr(config, f.name), Path) else v
                                   for f in dataclasses.fields(config)})
    conn = None
    with (folder / 'wire.jsonl').open('w', encoding='utf-8') as wire:
        def recorder(event):
            # Provider defines wire schema; retain its event transparently.
            wire.write(json.dumps(_redact(event, secrets), ensure_ascii=False) + '\n')
            wire.flush()
            os.fsync(wire.fileno())
        try:
            if metadata:
                _backup(Path(context['snapshot']) / 'history.sqlite', config.db_path, frozen=True)
            conn = db.connect(config.db_path)
            db.init_db(conn)
            conn.execute('DELETE FROM maint_queue')
            if selection is not None:
                conversation = selection['conversation_id']
            else:
                conversation = '__evaluation__'
                while conn.execute('SELECT 1 FROM chat_log WHERE instance=? AND conversation_id=? LIMIT 1',
                                   (config.instance, conversation)).fetchone():
                    conversation += '_'
                for row in context['recent_seed']:
                    conn.execute('INSERT INTO chat_log(created_at,instance,conversation_id,role,content,sources_json) '
                                 'VALUES(?,?,?,?,?,?)', (row['created_at'], config.instance, conversation,
                                                       row['role'], row['content'], '[]'))
            conn.commit()
            published = Published(True, text=context['text'])
            if metadata:
                fields = dict(metadata['published'])
                fields.update(text=context['text'], summary_path=str(Path(context['snapshot']) / 'summary.txt'),
                              memory_path=None)
                published = Published(**fields)
            if context.get('inject_memory') is False:
                published = Published(False, user_id=config.user_id,
                                      reason='memory context injection disabled for evaluation')
            from .providers import create_client
            client = create_client(config, case['model'], recorder=recorder, role='main')
            retrieval = (create_client(config, suite['retrieval_profile'], recorder=recorder, role='retrieval')
                         if context['mode'] == 'pipeline' else None)
            response = chat.chat_agentic(
                config, conn, case['question']['text'], published=published,
                instance=config.instance, conversation_id=conversation, maintenance=None,
                stream=False, client=client, retrieval_client=retrieval,
                now_epoch=suite['now_epoch'], allow_memory_tools=context['mode'] == 'pipeline',
                **chat_options)
            result.update(answer=response['answer'], tool_trace=response['tool_trace'], chat=response)
            result['checks'] = _checks(response['answer'], case['question'].get('checks', {}))
            result['status'] = 'passed' if all(c['passed'] for c in result['checks']) else 'failed'
            if after_trial is not None:
                try:
                    result['summary'] = after_trial(config, conn, response, case)
                except Exception as exc:
                    result['summary'] = {'status': 'failed', 'errors': [{
                        'type': type(exc).__name__[:80],
                        'message': 'summary maintenance failed; chat answer and checks retained'}]}
        except Exception as exc:
            # Do not persist arbitrary exception strings, even if credentials are
            # echoed by transport. Provider recorder retains its safe error data.
            result['error'] = {'type': type(exc).__name__[:80], 'message': 'trial failed; see sanitized wire events'}
        finally:
            if conn is not None:
                conn.close()
            if config.usage_file.exists():
                result['usage'] = _read(config.usage_file)
            result['elapsed_seconds'] = time.perf_counter() - started
            result = _redact(result, secrets)
            _write(folder / 'result.json', result)
    return result


def compare_runs(records: list[dict], previous: list[dict]) -> dict:
    current = {tuple(r['identity']): r for r in records}
    old = {tuple(r['identity']): r for r in previous}
    shared = current.keys() & old.keys()
    return {'new': [list(k) for k in sorted(current.keys() - old.keys())],
            'missing': [list(k) for k in sorted(old.keys() - current.keys())],
            'status_changes': [{'identity': list(k), 'before': old[k]['status'], 'after': current[k]['status']}
                               for k in sorted(shared) if old[k]['status'] != current[k]['status']],
            'new_failures': [list(k) for k in sorted(shared)
                             if old[k]['status'] == 'passed' and current[k]['status'] != 'passed'],
            'resolved_failures': [list(k) for k in sorted(shared)
                                  if old[k]['status'] != 'passed' and current[k]['status'] == 'passed']}


def _reports(run: Path, cases: list[dict], records: list[dict], previous: list[dict]) -> None:
    records = sorted(records, key=lambda r: r['index'])
    _write(run / 'results.json', records)
    regression = compare_runs(records, previous)
    _write(run / 'regression.json', regression)
    note = ('Automatic literal checks are not a human quality score. Synthetic fixtures test mechanics only. '
            'Fixed input modes freeze supplied text; pipeline models may choose different queries and handoffs '
            'despite the same retrieval profile. API outputs are nondeterministic.')
    columns = sorted({(c['model']['id'], c['context']['id']) for c in cases})
    found = {tuple(r['identity']): r for r in records}
    rows = []
    blind = ['# Blind answer comparison', '', note, '']
    blind_key = {}
    for question, repeat in dict.fromkeys((c['question']['id'], c['repeat']) for c in cases):
        q = next(c['question']['text'] for c in cases if c['question']['id'] == question)
        cells = []
        blind.extend([f'## {question} / repeat {repeat}', '', q, ''])
        for model, context in columns:
            identity = (model, question, context, repeat)
            record = found.get(identity)
            case = next(c for c in cases if tuple(c['identity']) == identity)
            blind_id = 'B-' + case['case_id']
            blind_key[blind_id] = list(identity)
            if record:
                status = record['status']
                text = record['answer'] or _json(record['error'])
                stats = f'{status}; {record["elapsed_seconds"]:.3f}s; usage {_json(record["usage"])}'
                cells.append('<td><pre>' + html.escape(stats) + '</pre><pre>' + html.escape(text) + '</pre></td>')
                blind.extend([f'### {blind_id} ({status})', '', text, ''])
            else:
                cells.append('<td>pending / not scheduled</td>')
        rows.append('<tr><th><pre>' + html.escape(f'{question} / {repeat}\n{q}') + '</pre></th>' + ''.join(cells) + '</tr>')
    page = ('<!doctype html><meta charset="utf-8"><title>Evaluation comparison</title>'
            '<style>body{font:15px system-ui;margin:24px}table{border-collapse:collapse;width:100%}'
            'td,th{border:1px solid #bbb;vertical-align:top;padding:12px;min-width:240px}'
            'pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}</style>'
            '<h1>Evaluation comparison</h1><p>' + html.escape(note) + '</p>'
            f'<p>Completed {len(records)} / {len(cases)}; errors {sum(r["status"] == "error" for r in records)}; '
            f'failed checks {sum(r["status"] == "failed" for r in records)}</p>'
            '<table><tr><th>Question / repeat</th>'
            + ''.join('<th>' + html.escape(m + ' / ' + c) + '</th>' for m, c in columns)
            + '</tr>' + ''.join(rows) + '</table><h2>Regression</h2><pre>' + html.escape(_json(regression)) + '</pre>')
    (run / 'comparison.html').write_text(page, encoding='utf-8')
    (run / 'blind.md').write_text('\n'.join(blind), encoding='utf-8')
    _write(run / 'blind-key.json', blind_key)


def run_suite(path: Path | str, *, out: str | Path | None = None,
              previous: Path | str | None = None, evaluation_root: Path | None = None,
              after_trial=None) -> Path:
    loaded = load_suite(path)
    plan = plan_suite(loaded)
    if not plan['valid']:
        raise EvaluationError('missing environment credentials for profiles: ' +
                              ', '.join(p['profile'] for p in plan['missing_credentials']))
    suite = loaded['suite']
    previous_path = _input(previous) if previous else None
    previous_records = _read(previous_path / 'results.json') if previous_path else []
    sources = (loaded['path'], *(Path(p) for p in loaded['snapshots']),
               *(Path(p) for p in loaded['files']), *((previous_path,) if previous_path else ()))
    run = _destination('run', out, evaluation_root, sources)
    (run / 'inputs').mkdir()
    (run / 'trials').mkdir()
    (run / 'inputs' / 'suite.json').write_bytes(loaded['raw'])
    inputs = {'inputs/suite.json': _hash(loaded['raw'])}
    for index, data in enumerate(loaded['files'].values()):
        name = f'inputs/context-{index}.txt'
        (run / name).write_bytes(data)
        inputs[name] = _hash(data)
    frozen = {}
    for index, (source, metadata) in enumerate(loaded['snapshots'].items()):
        target = run / 'inputs' / f'snapshot-{index}'
        target.mkdir()
        for filename in ('history.sqlite', 'summary.txt', 'snapshot.json'):
            data_path = _input(Path(source) / filename)
            # Copy exact snapshot bytes; trial uses SQL backup of this run copy.
            shutil.copyfile(data_path, target / filename)
            inputs[str((target / filename).relative_to(run))] = _file_hash(target / filename)
        frozen[str(target)] = _snapshot(target, suite['now_epoch'])
        if frozen[str(target)] != metadata:
            raise EvaluationError('snapshot manifest changed while freezing experiment')
        for context in suite['contexts']:
            if context.get('snapshot') == source:
                if context['text'] != (target / 'summary.txt').read_bytes().decode('utf-8'):
                    raise EvaluationError('snapshot summary changed while freezing experiment')
                context['snapshot'] = str(target)
    if previous_path:
        _write(run / 'inputs' / 'previous-results.json', previous_records)
        inputs['inputs/previous-results.json'] = _file_hash(run / 'inputs' / 'previous-results.json')
    replay = json.loads(_json(suite))
    replay.pop('recent_seed', None)
    for context in replay['contexts']:
        if 'recent_selection' in context:
            context['recent_seed'] = context.pop('recent_selection')
        if context['mode'] == 'none':
            context.pop('text')
            context.pop('recent_seed')
        elif context['mode'] in ('snapshot', 'pipeline'):
            context.pop('text')
            context['snapshot'] = Path(context['snapshot']).name
    _write(run / 'inputs' / 'materialized-suite.json', replay)
    inputs['inputs/materialized-suite.json'] = _file_hash(run / 'inputs' / 'materialized-suite.json')
    cases = build_cases(suite)
    code = {str(p.relative_to(PROJECT_ROOT)): _file_hash(p)
            for p in sorted((PROJECT_ROOT / 'src' / 'continuity').glob('*.py'))}
    if (PROJECT_ROOT / 'eval.py').exists():
        code['eval.py'] = _file_hash(PROJECT_ROOT / 'eval.py')
    manifest = {'version': 1, 'status': 'running', 'inputs': inputs, 'code_sha256': code,
                'fixed_now_epoch': suite['now_epoch'], 'suite_sha256': _hash(loaded['raw']),
                'cases': plan['cases'], 'budgets': suite['budgets'], 'chat_options': suite.get('chat', {}),
                'note': 'No maintenance workers. See comparison.html for fixed-input vs pipeline semantics.'}
    _write(run / 'manifest.json', manifest)
    profiles = suite['models'] + ([suite['retrieval_profile']] if 'retrieval_profile' in suite else [])
    secrets = tuple({os.environ[p['api_key_env']] for p in profiles
                     if p.get('api_key_env') and os.environ.get(p['api_key_env'])})
    records = []
    executor = ThreadPoolExecutor(max_workers=suite['concurrency'])
    pending = {}
    iterator = iter(cases)
    interrupted = False
    with (run / 'results.jsonl').open('w', encoding='utf-8') as results:
        def save(record):
            records.append(record)
            results.write(json.dumps(record, ensure_ascii=False) + '\n')
            results.flush()
            os.fsync(results.fileno())
            _reports(run, cases, records, previous_records)
        def schedule():
            case = next(iterator, None)
            if case is not None:
                pending[executor.submit(_trial, run, case, suite, frozen, secrets, after_trial)] = case
        try:
            for _ in range(min(suite['concurrency'], len(cases))):
                schedule()
            while pending:
                done, _ = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
                for future in sorted(done, key=lambda f: pending[f]['index']):
                    pending.pop(future)
                    save(future.result())
                    schedule()
        except KeyboardInterrupt:
            interrupted = True
            # Stop scheduling; bounded in-flight calls finish under profile timeout.
            for future in pending:
                future.cancel()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            for future in pending:
                if not future.cancelled():
                    save(future.result())
            _reports(run, cases, records, previous_records)
            manifest.update(status='interrupted' if interrupted else 'complete', completed=len(records))
            _write(run / 'manifest.json', manifest)
    return run


def resume_run(run: Path | str, *, concurrency=16, after_trial=None) -> Path:
    """Resume frozen cases, retaining completed results and interrupted spending."""
    _number(concurrency, integer=True, minimum=1)
    if concurrency > 16:
        raise EvaluationError('resume concurrency must be between 1 and 16')
    run = _input(run)
    manifest = _read(run / 'manifest.json')
    for name, digest in manifest['inputs'].items():
        if _file_hash(_input(run / name)) != digest:
            raise EvaluationError('frozen run input hash mismatch')
    loaded = load_suite(run / 'inputs' / 'materialized-suite.json')
    suite = loaded['suite']
    cases = build_cases(suite)
    if manifest['cases'] != [{'case_id': c['case_id'], 'identity': c['identity']} for c in cases]:
        raise EvaluationError('frozen run case identities changed')
    plan = plan_suite(loaded)
    if not plan['valid']:
        raise EvaluationError('missing environment credentials for profiles: ' +
                              ', '.join(p['profile'] for p in plan['missing_credentials']))
    previous_path = run / 'inputs' / 'previous-results.json'
    previous_records = _read(previous_path) if previous_path.exists() else []
    records = []
    completed = set()
    by_id = {c['case_id']: c for c in cases}

    def retain(record):
        case_id = record['case_id']
        if case_id not in by_id or record['identity'] != by_id[case_id]['identity']:
            raise EvaluationError('completed result does not match frozen cases')
        if case_id not in completed:
            records.append(record)
            completed.add(case_id)

    results_path = run / 'results.jsonl'
    data = results_path.read_bytes() if results_path.exists() else b''
    boundary = data.rfind(b'\n') + 1
    for line in data[:boundary].splitlines():
        if line.strip():
            retain(json.loads(line))
    unindexed = []
    for case in cases:
        result_path = run / 'trials' / case['case_id'] / 'result.json'
        if case['case_id'] not in completed and result_path.exists():
            record = _read(result_path)
            retain(record)
            unindexed.append(record)
    unfinished = [c for c in cases if c['case_id'] not in completed]
    initial_usage = {}
    archives = run / 'interrupted-attempts'
    for case in unfinished:
        folder = run / 'trials' / case['case_id']
        if not folder.exists():
            continue
        usage_path = folder / 'usage.json'
        if usage_path.exists():
            usage = _read(usage_path)
            if not isinstance(usage, dict):
                raise EvaluationError('interrupted trial usage must be an object')
            initial_usage[case['case_id']] = usage
        summary_usage = folder / 'state' / 'ledger' / 'usage.json'
        summary_state = folder / 'state' / 'sandbox-summary.json'
        if summary_usage.exists() and summary_state.exists():
            spent = _read(summary_usage)
            baseline = _read(summary_state)['baseline']
            case['_resume_summary_usage'] = {
                key: max(0, value - baseline.get(key, 0)) for key, value in spent.items()}
    history = manifest.setdefault('resume_history', [])
    history.append({
        'resumed_at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'old_concurrency': history[-1]['new_concurrency'] if history else suite['concurrency'],
        'new_concurrency': concurrency, 'completed_before': len(records),
    })
    manifest.update(status='running', completed=len(records))
    _write(run / 'manifest.json', manifest)
    for case in unfinished:
        folder = run / 'trials' / case['case_id']
        if folder.exists():
            archives.mkdir(exist_ok=True)
            folder.rename(archives / f'{case["case_id"]}-{uuid.uuid4().hex}')
    if boundary < len(data):
        archives.mkdir(exist_ok=True)
        (archives / f'results-{uuid.uuid4().hex}.jsonl').write_bytes(data[boundary:])
        with results_path.open('r+b') as results:
            results.truncate(boundary)
            results.flush()
            os.fsync(results.fileno())
    profiles = suite['models'] + ([suite['retrieval_profile']] if 'retrieval_profile' in suite else [])
    secrets = tuple({os.environ[p['api_key_env']] for p in profiles
                     if p.get('api_key_env') and os.environ.get(p['api_key_env'])})
    executor = ThreadPoolExecutor(max_workers=concurrency)
    pending = {}
    iterator = iter(unfinished)
    interrupted = False
    with results_path.open('a', encoding='utf-8') as results:
        def append(record):
            results.write(json.dumps(record, ensure_ascii=False) + '\n')
            results.flush()
            os.fsync(results.fileno())
        def save(record):
            retain(record)
            append(record)
            _reports(run, cases, records, previous_records)
        def schedule():
            case = next(iterator, None)
            if case is not None:
                future = executor.submit(_trial, run, case, suite, loaded['snapshots'], secrets,
                                         after_trial, initial_usage=initial_usage.get(case['case_id']))
                pending[future] = case
        try:
            for record in unindexed:
                append(record)
            _reports(run, cases, records, previous_records)
            for _ in range(min(concurrency, len(unfinished))):
                schedule()
            while pending:
                done, _ = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
                for future in sorted(done, key=lambda f: pending[f]['index']):
                    pending.pop(future)
                    save(future.result())
                    schedule()
        except KeyboardInterrupt:
            interrupted = True
            for future in pending:
                future.cancel()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            try:
                for future in pending:
                    if not future.cancelled():
                        save(future.result())
            finally:
                manifest.update(status='interrupted' if interrupted or len(records) != len(cases)
                                else 'complete', completed=len(records))
                _write(run / 'manifest.json', manifest)
    return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    snapshot = commands.add_parser('snapshot', help='read selected app summary/history into a NEW frozen snapshot')
    snapshot.add_argument('--out', help='new path inside app/evaluations')
    snapshot.add_argument('--empty-summary', action='store_true', help='intentionally select empty summary, without fallback')
    snapshot.add_argument('--recent-conversation')
    snapshot.add_argument('--recent-limit', type=int, default=40)
    for name in ('plan', 'run'):
        command = commands.add_parser(name)
        command.add_argument('suite', type=Path)
        if name == 'run':
            command.add_argument('--out')
            command.add_argument('--previous', type=Path)
            command.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == 'snapshot':
            config = load_config()
            recent = ({'instance': config.instance, 'conversation_id': args.recent_conversation,
                       'limit': args.recent_limit} if args.recent_conversation else None)
            print(create_snapshot(source_config=config, out=args.out, empty_summary=args.empty_summary, recent=recent))
        elif args.command == 'plan' or args.dry_run:
            plan = plan_suite(load_suite(args.suite))
            print(_json(plan))
            return 0 if plan['valid'] else 2
        else:
            print(run_suite(args.suite, out=args.out, previous=args.previous))
        return 0
    except EvaluationError as exc:
        print('Evaluation rejected: ' + str(exc)[:400], file=sys.stderr)
        return 2
    except (OSError, ValueError, sqlite3.Error, TypeError, KeyError):
        # Never echo arbitrary input/provider bodies in terminal error handling.
        print('Evaluation rejected: invalid input, output path, snapshot, or missing credentials. '
              'Use plan to inspect credential references.', file=sys.stderr)
        return 2
