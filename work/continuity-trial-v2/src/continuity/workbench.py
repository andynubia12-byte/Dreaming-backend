"""Loopback workbench; formal configuration is a read-only input, never runtime state.

All writes live below evaluations/workbench (or the explicit factory test root).
No maintenance queue, import worker, credential discovery or provider calls at boot.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import re
import socket
import threading
import time
import uuid
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from . import chat, db, evaluation
from .config import Config, load_config
from .events import Observer, observe
from .model_pool import ModelPool
from .providers import Profile, create_client, redact_record
from .summary_store import Published, read_published
from .workbench_memory_v1 import WorkbenchMemoryV1, is_v1

MAX_BODY = 32 * 1024 * 1024
ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z')


class HTTPError(ValueError):
    def __init__(self, message='请求格式无效。', status=400):
        super().__init__(message)
        self.status = status


def keys(value, allowed, required=()):
    if not isinstance(value, dict) or set(value) - set(allowed) or set(required) - set(value):
        raise HTTPError('字段缺失或含不支持的字段。')


def identifier(value):
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise HTTPError('无效的标识。')
    return value


def read(path):
    return json.loads(path.read_text('utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def append(path, value):
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o600)


def jsonlines(path):
    if not path.exists():
        return []
    result = []
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                # A concurrent append or crash may leave one unfinished final line.
                break
    return result


def config_json(config):
    return {field.name: str(value) if isinstance(value := getattr(config, field.name), Path) else value
            for field in dataclasses.fields(config)}


def restore_config(value):
    paths = {'project_root', 'db_path', 'runs_dir', 'maint_state_dir', 'usage_file',
             'maint_cli', 'export_dir', 'retrieval_source_db', 'retrieval_manifest'}
    return Config(**{k: Path(v) if k in paths and v is not None else v for k, v in value.items()})


def summary_options(value, pool=None):
    """Resolve one detached Responses profile; model always matches that profile."""
    value = copy.deepcopy(value if value is not None else {'enabled': False})
    keys(value, {'engine', 'enabled', 'model', 'profile', 'max_requests', 'max_output_tokens'}, {'enabled'})
    if value.get('engine', 'v2.6') not in ('v1', 'v2.6'):
        raise HTTPError('summary.engine 必须为 v1 或 v2.6。')
    value.setdefault('engine', 'v2.6')
    if type(value['enabled']) is not bool:
        raise HTTPError('summary.enabled 必须为布尔值。')
    selected = value.get('profile')
    requested = value.get('model')
    if selected is None:
        if pool is not None:
            profiles = pool.profiles()
            selected = next((p for p in profiles if p['id'] == (requested or 'deepseek-flash')), None)
            if selected is None:
                selected = next((p for p in profiles if p['model'] == requested), None)
            if selected is None:
                raise HTTPError('摘要模型不在共享模型池中。')
        else:
            if requested not in (None, 'deepseek-flash'):
                raise HTTPError('摘要模型需要完整 Responses 配置或共享模型池。')
            selected = {'id': 'deepseek-flash', 'model': 'deepseek-flash',
                'base_url': 'https://api.deepseek.com', 'api_format': 'responses',
                'api_key_env': 'DEEPSEEK_API_KEY', 'parameters': {
                    'reasoning': {'effort': 'high'}, 'max_output_tokens': 64000}}
    selected = profile(selected)
    if 'profile' in value and requested is not None and requested != selected['model']:
        raise HTTPError('summary.model 必须与 summary.profile.model 一致。')
    value.update(profile=selected, model=selected['model'])
    for field, default, maximum in [('max_requests', 30, 1000), ('max_output_tokens', 64000, 64000)]:
        value.setdefault(field, default)
        if (type(value[field]) is not int or not 0 <= value[field] <= maximum
                or field == 'max_output_tokens' and (value[field] < 1 or
                    value.get('engine') != 'v1' and value[field] != 64000)):
            raise HTTPError('摘要预算超出允许范围。')
    if value.get('engine') == 'v1' and value['max_output_tokens'] < 16000:
        raise HTTPError('v1 摘要至少需要 16000 output tokens。')
    return value


def runtime_profile(value, migrations, role):
    """Explicit legacy read boundary; new browser inputs never use this path."""
    value = copy.deepcopy(value)
    if isinstance(value, dict) and value.get('api_format') == 'chat_completions':
        from .providers import migrate_legacy_profile
        value.pop('label', None)
        value.pop('builtin', None)
        value = migrate_legacy_profile(value)
        migrations.append({'role': role, 'id': value['id'], 'from': 'chat_completions',
                           'to': 'responses', 'scope': 'runtime_copy'})
    return profile(value)


def profile(value):
    value = copy.deepcopy(value)
    if isinstance(value, dict):
        value.pop('label', None)
        value.pop('builtin', None)
    try:
        if not isinstance(value, dict) or value.get('api_format') != 'responses':
            raise ValueError('Responses API is required')
        evaluation._profile(value)
        Profile(value)
    except (ValueError, TypeError, KeyError):
        raise HTTPError('模型配置无效；仅支持 Responses API，凭据只能填写环境变量引用或订阅认证类型。') from None
    return value


def profile_fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def connectivity_outcome(event):
    """Explain recorded evidence; credentials alone never imply connectivity."""
    status = event.get('http_status')
    response = event.get('response') or {}
    upstream = response.get('error') if isinstance(response, dict) else None
    if not upstream:
        upstream = next((e.get('error') or e for e in reversed(event.get('events', []))
                         if e.get('type') in ('error', 'response.error')), {})
    if not isinstance(upstream, dict):
        upstream = {'message': upstream}
    error = {name: str(upstream[name])[:800] for name in ('code', 'param', 'message')
             if upstream.get(name) is not None}
    diagnostic = event.get('error') or {}
    finish = event.get('finish_reason')
    protocol = response.get('status') if isinstance(response, dict) else None
    protocol = protocol or ('http_error' if status is not None and status >= 400 else
        'invalid_stream' if diagnostic.get('type') == 'StreamProtocolError' else 'not_received')
    category = 'protocol'
    if status in (401, 403) or diagnostic.get('type') == 'MissingCredential':
        category = 'auth'
    elif status == 404:
        category = 'model' if 'model' in str(error).lower() else 'not_found'
    elif status in (400, 422):
        category = 'model' if (error.get('param') == 'model' or
            'model_not_found' in error.get('code', '').lower()) else 'parameter'
    elif status == 429:
        category = 'rate_limit'
    elif status is not None and status >= 500:
        category = 'upstream'
    elif status is None and event.get('dispatch_attempted'):
        category = 'network'
    elif protocol == 'incomplete':
        category = 'incomplete'
    elif protocol == 'completed' and finish == 'stop':
        category = 'success'
    sample = ''
    if isinstance(response, dict):
        output = response.get('output')
        sample = ''.join(str(part.get('text') or '') for item in output if isinstance(item, dict)
            and isinstance(item.get('content'), list) for part in item['content']
            if isinstance(part, dict) and part.get('type') == 'output_text')[:800] if isinstance(output, list) else ''
    message = error.get('message') or {
        'success': '基础流式对话通过；工具、JSON、长上下文和摘要兼容性未测试。',
        'incomplete': '已收到响应，但输出不完整；保留原参数与推理强度，未自动降级。',
    }.get(category) or (response.get('error_body_text') if isinstance(response, dict) else None) or diagnostic.get('reason') or '未获得完整的 Responses 流响应。'
    return {'ok': category == 'success', 'category': category, 'http_status': status,
            'model': event.get('model'), 'url': event.get('url'),
            'protocol_outcome': protocol, 'finish_reason': finish, 'error': error,
            'incomplete_details': response.get('incomplete_details') if isinstance(response, dict) else None,
            'message': message[:800], 'response_sample': sample}


def isolated(source, folder, model, retrieval):
    config = dataclasses.replace(source, project_root=folder, db_path=folder / 'chat.sqlite',
        runs_dir=folder / 'runs', maint_state_dir=folder / 'state', usage_file=folder / 'usage.json',
        maint_cli=folder / 'disabled-maintenance.py', export_dir=None,
        retrieval_source_db=None, retrieval_manifest=None, instance=source.instance,
        main_chat_model=model['model'], chat_model=retrieval['model'],
        api_base_url=model['base_url'], api_key_env=model.get('api_key_env', 'WORKBENCH_UNUSED_KEY'))
    for field in dataclasses.fields(config):
        value = getattr(config, field.name)
        if isinstance(value, Path) and not value.is_relative_to(folder):
            raise HTTPError('隔离配置含外部写路径。')
    config.ensure_dirs()
    write(folder / 'config.json', config_json(config))
    return config


def memory_json(published):
    return {'text': published.text if published.present else '', 'version': published.version,
            'present': published.present, 'chars': len(published.text) if published.present else 0,
            'published_at': published.published_at, 'now_max': published.now_max}


class RunEvents:
    """Disk is the replay registry; the condition only wakes live subscribers."""
    def __init__(self, folder, run):
        self.folder, self.run = folder, run
        self.condition = threading.Condition()
        event_path = folder / 'events.jsonl'
        event_path.touch(mode=0o600, exist_ok=True)
        # Scan without retaining past contexts; only a torn last append is removed.
        self.seq = 0
        self.last_phase = None
        with event_path.open('rb+') as handle:
            while True:
                position = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.endswith(b'\n'):
                    handle.truncate(position)
                    break
                event = json.loads(line)
                self.seq, self.last_phase = event['seq'], event.get('phase')
        self.secrets = ()

    def publish(self, event):
        if event.get('phase') == 'model_thinking':
            event = {**event, 'data': {'status': 'thinking'}}
        safe = redact_record(event, self.secrets)
        with self.condition:
            self.seq += 1
            safe.update(seq=self.seq, time=time.time(), root_run_id=self.run['id'])
            append(self.folder / 'events.jsonl', safe)
            self.last_phase = safe.get('phase')
            self.condition.notify_all()
        return safe


class Workbench:
    def __init__(self, source_config, root):
        self.source = source_config
        self.root = Path(root).absolute()
        # The root is operator supplied, never supplied by the browser.
        evaluation._no_symlinks(self.root)
        for source in (source_config.db_path, source_config.maint_state_dir, source_config.runs_dir):
            if evaluation._overlap(self.root, source):
                raise ValueError('workbench root overlaps formal data')
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        for name in ('sessions', 'runs', 'batches', 'suites', 'snapshots', 'batch-inputs'):
            (self.root / name).mkdir(exist_ok=True)
        self.guard = threading.RLock()
        self.sessions, self.runs, self.batches, self.hubs, self.locks = {}, {}, {}, {}, {}
        self.snapshots, self.memories = {}, {}
        defaults = [
            {'id': 'deepseek-pro', 'model': source_config.main_chat_model,
             'base_url': source_config.api_base_url, 'api_format': 'responses',
             'api_key_env': source_config.api_key_env, 'parameters': {'max_output_tokens': 64000, 'reasoning': {'effort': 'high'}}},
            {'id': 'deepseek-flash', 'model': source_config.chat_model,
             'base_url': source_config.api_base_url, 'api_format': 'responses',
             'api_key_env': source_config.api_key_env, 'parameters': {'max_output_tokens': 64000, 'reasoning': {'effort': 'high'}}},
            {'id': 'codex-sol-high', 'model': 'gpt-6.1-sol',
             'base_url': 'https://chatgpt.com/backend-api/codex', 'api_format': 'responses',
             'auth': 'codex_subscription', 'parameters': {'temperature': None,
             'reasoning': {'effort': 'high'}, 'max_output_tokens': 32000}},
        ]
        self.pool = ModelPool(self.root, defaults)
        self.builtin_model_ids = frozenset(item['id'] for item in defaults)
        self.default_model_id = defaults[0]['id']
        self.retrieval_model_id = defaults[1]['id']
        self.memory_v1 = WorkbenchMemoryV1(self)
        self._recover()

    def _recover(self):
        for file in (self.root / 'sessions').glob('*/session.json'):
            record = read(file)
            identifier(record['id'])
            if record.get('deleted_at') is not None:
                continue
            self.sessions[record['id']] = record
            workspace = record.get('workspace_id', record['id'])
            self.locks[record['id']] = self.locks.setdefault(workspace, threading.Lock())
        for file in (self.root / 'runs').glob('*/run.json'):
            run = read(file)
            self.runs[identifier(run['id'])] = run
            hub = RunEvents(file.parent, run)
            self.hubs[run['id']] = hub
            if run['status'] in ('running', 'accepted'):
                run.update(status='interrupted', finished_at=time.time())
                if run.get('kind') == 'memory_v1':
                    self.memory_v1.recover(run)
                elif (file.parent / 'summary-request.json').exists():
                    self.memory_v1.recover_chat(run, file.parent)
                write(file, run)
                hub.publish({'phase': 'run_complete', 'session_id': run['session_id'],
                             'request_id': run['request_id'], 'run_id': run['id'],
                             'data': {'status': run['status'], 'result': run.get('result'),
                                      'summary': run.get('summary'),
                                      'error': run.get('error') if run.get('recovered') else
                                               run.get('error', '服务重启；未自动重发请求。'),
                                      'error_type': run.get('error_type')}})
            elif hub.last_phase != 'run_complete':
                # A crash may occur after run.json is durable but before its final SSE append.
                hub.publish({'phase': 'run_complete', 'session_id': run['session_id'],
                    'request_id': run['request_id'], 'run_id': run['id'], 'data': {
                        'status': run['status'], 'result': run.get('result'), 'summary': run.get('summary'),
                        'error': run.get('error'), 'error_type': run.get('error_type')}})
        for file in (self.root / 'batches').glob('*/batch.json'):
            batch = read(file)
            if batch['status'] in ('running', 'accepted'):
                manifest = file.parent / 'run' / 'manifest.json'
                status = read(manifest).get('status') if manifest.exists() else None
                batch['status'] = 'completed' if status == 'complete' else 'interrupted'
                write(file, batch)
            self.batches[identifier(batch['id'])] = batch

    def session(self, sid):
        identifier(sid)
        with self.guard:
            if sid not in self.sessions:
                raise HTTPError('会话不存在。', 404)
            return self.runtime_session(self.sessions[sid])

    def runtime_summary(self, value, migrations):
        value = copy.deepcopy(value)
        if 'profile' in value:
            value['profile'] = runtime_profile(value['profile'], migrations, 'summary')
        else:
            migrations.append({'role': 'summary', 'from': 'legacy_model_only',
                               'to': 'responses', 'scope': 'runtime_copy'})
        return summary_options(value, self.pool)

    def runtime_session(self, stored):
        session = copy.deepcopy(stored)
        migrations = session.setdefault('protocol_migrations', [])
        session['model'] = runtime_profile(session['model'], migrations, 'main')
        session['retrieval']['profile'] = runtime_profile(session['retrieval']['profile'], migrations, 'retrieval')
        session['summary'] = self.runtime_summary(session['summary'], migrations)
        return session

    def discover(self):
        current = read_published(self.source.maint_state_dir, self.source.user_id)
        memories = {'current': {'id': 'current', 'label': '当前已发布摘要', 'source_kind': 'published',
                                **memory_json(current)}}
        old = self.root / 'saved-comparisons' / 'v65.md'
        if old.is_file() and not old.is_symlink():
            text = old.read_text('utf-8')
            memories['v65'] = {'id': 'v65', 'label': 'v65 原样摘要', 'source_kind': 'saved_comparison',
                               'version': 65, 'chars': len(text), 'text': text}
        snapshots = {}
        folders = [*(self.source.project_root / 'evaluations').glob('snapshot-*'),
                   *(self.root / 'snapshots').glob('snapshot-*')]
        for folder in folders:
            try:
                evaluation._no_symlinks(folder)
                meta = read(folder / 'snapshot.json')
                summary = (folder / 'summary.txt').read_bytes()
                if (meta.get('version') != 1 or not (folder / 'history.sqlite').is_file()
                        or hashlib.sha256(summary).hexdigest() != meta.get('summary_sha256')):
                    continue
                sid = identifier(folder.name)
                snapshots[sid] = folder
                memories[sid] = {'id': sid, 'label': f'冻结摘要 v{meta["published"]["version"]}',
                    'version': meta['published']['version'], 'chars': len(summary.decode('utf-8')),
                    'source_kind': 'snapshot'}
            except (OSError, ValueError, KeyError):
                continue
        with self.guard:
            self.memories, self.snapshots = memories, snapshots
        return current

    def bootstrap(self):
        self.discover()
        with self.guard:
            return {**self.models(), 'memories': list(self.memories.values()),
                'snapshots': [self.memories[sid] for sid in self.snapshots],
                'defaults': {'memory_id': 'current', 'model_id': self.default_model_id,
                             'retrieval_profile': self.retrieval_profile,
                             'summary_profile': summary_options(None, self.pool)},
                'sessions': [{k: s[k] for k in ('id', 'workspace_id', 'title', 'created_at') if k in s}
                             for s in self.sessions.values()],
                'batches': [{k: b[k] for k in ('id', 'status', 'created_at', 'finished_at') if k in b}
                            for b in self.batches.values()],
                'server_time': time.time(), 'capabilities': {
                    'chat': True, 'batch': True, 'replay': True, 'independent_agents': True,
                    'custom_profiles': True, 'model_availability': 'unknown', 'batch_summary': True, 'max_concurrency': 16,
                    'api_format': 'responses', 'canonical_input': True,
                    'summary': {'api_format': 'responses', 'model_pool': True,
                        'budget_scope': 'cumulative_per_workspace', 'additional_to_chat_budget': True},
                    'subscription_auth': 'unchecked_until_explicit_run'}}

    @property
    def retrieval_profile(self):
        selected = next(p for p in self.pool.profiles() if p['id'] == self.retrieval_model_id)
        return {k: v for k, v in selected.items() if k != 'label'}

    def models(self):
        with self.guard:
            return {'profiles': [{**p, 'builtin': p['id'] in self.builtin_model_ids}
                                 for p in self.pool.profiles()],
                    'credential_status': self.pool.credential_status(),
                    'protocol_migrations': self.pool.protocol_migrations()}

    def save_model(self, body):
        keys(body, {'profile', 'label', 'api_key'}, {'profile'})
        key = body.get('api_key')
        if isinstance(key, str) and key.strip():
            encoded = json.dumps(key.strip(), ensure_ascii=False)[1:-1]
            if encoded in json.dumps([body['profile'], body.get('label')], ensure_ascii=False):
                raise HTTPError('密钥不能出现在模型配置或名称中。')
        with self.guard:
            submitted = copy.deepcopy(body['profile'])
            if isinstance(submitted, dict):
                submitted.pop('builtin', None)
            saved = self.pool.save(submitted, label=body.get('label'), api_key=body.get('api_key'))
            return {'profile': {**saved, 'builtin': saved['id'] in self.builtin_model_ids},
                    'credential_status': self.pool.credential_status()}

    def test_model(self, body):
        keys(body, {'model_id', 'profile', 'api_key'})
        if ('model_id' in body) == ('profile' in body) or 'model_id' in body and 'api_key' in body:
            raise HTTPError('请提交已保存的模型 ID，或待测试的模型配置及可选临时 Key。')
        selected = copy.deepcopy(self.pool.get(body['model_id']) if 'model_id' in body else body['profile'])
        temporary_key = body.get('api_key')
        reference = None
        if temporary_key is not None:
            if (not isinstance(temporary_key, str) or not 1 <= len(temporary_key.strip()) <= 8192
                    or any(not 32 <= ord(char) <= 126 for char in temporary_key)
                    or not isinstance(selected, dict) or selected.get('auth') == 'codex_subscription'):
                raise HTTPError('临时 Key 无效，或当前认证方式不接受临时 Key。')
            temporary_key = temporary_key.strip()
            if json.dumps(temporary_key, ensure_ascii=False)[1:-1] in json.dumps(selected, ensure_ascii=False):
                raise HTTPError('密钥不能出现在模型配置或名称中。')
            reference = 'WORKBENCH_CONNECTIVITY_KEY_' + uuid.uuid4().hex.upper()
            while reference in os.environ:
                reference = 'WORKBENCH_CONNECTIVITY_KEY_' + uuid.uuid4().hex.upper()
        fingerprint = profile_fingerprint(selected)
        if reference:
            selected['api_key_env'] = reference
        selected = profile(selected)
        selected['timeout_seconds'] = min(selected.get('timeout_seconds', 180), 30)
        test_id = uuid.uuid4().hex
        folder = self.root / 'connectivity' / test_id
        evaluation._no_symlinks(folder)
        config = isolated(self.source, folder, selected, selected)
        config = dataclasses.replace(config, max_requests=1, trial_max_requests=1,
                                     chat_max_output_tokens=4096, chat_final_max_output_tokens=4096)
        write(folder / 'config.json', config_json(config))
        recorded = []
        secrets = tuple(s for s in (temporary_key, os.environ.get(selected.get('api_key_env', ''))) if s)

        def recorder(event):
            # Transport records are already redacted. Also scrub property names and
            # redaction paths should an upstream echo a key there.
            encoded = json.dumps(event, ensure_ascii=False)
            for secret in secrets:
                encoded = encoded.replace(json.dumps(secret, ensure_ascii=False)[1:-1], '[REDACTED_CREDENTIAL]')
            safe = json.loads(encoded)
            recorded.append(safe)
            append(folder / 'wire.jsonl', safe)

        started = time.monotonic()
        try:
            if reference:
                os.environ[reference] = temporary_key
            client = create_client(config, selected, recorder=recorder, role='main')
            client.chat_stream([{'role': 'system', 'content': 'Reply OK'},
                {'role': 'user', 'content': 'Hi'}], on_event=lambda *args, **kwargs: None,
                max_tokens=4096, temperature=0.6)
        except Exception:
            # The transport recorder owns safe diagnostics, including upstream errors.
            pass
        finally:
            if reference:
                os.environ.pop(reference, None)
        event = recorded[-1] if recorded else {}
        outcome = connectivity_outcome(event)
        outcome.update(test_id=test_id, profile_fingerprint=fingerprint,
                       elapsed_seconds=round(time.monotonic() - started, 3),
                       request_count=client.usage.requests if recorded else 0,
                       artifacts=str(folder.relative_to(self.root)))
        write(folder / 'result.json', outcome)
        return outcome

    def delete_model(self, model_id):
        identifier(model_id)
        with self.guard:
            self.pool.delete(model_id)
            return self.models()

    def discover_models(self, body):
        from .model_discovery import discover_models
        return discover_models(self.source, self.pool, body)

    def rename_session(self, sid, body):
        from .session_management import rename_session
        return rename_session(self, sid, body)

    def delete_session(self, sid):
        from .session_management import delete_session
        return delete_session(self, sid)

    @staticmethod
    def public_session(session):
        return {k: v for k, v in session.items() if k not in ('initial', 'config')}

    def freeze(self, expected=None):
        before = read_published(self.source.maint_state_dir, self.source.user_id)
        if expected is not None and (before.version, before.text) != (expected.version, expected.text):
            raise HTTPError('当前摘要已变化，请重新选择。', 409)
        target = evaluation.create_snapshot(source_config=self.source,
            out=self.root / 'snapshots' / ('snapshot-' + uuid.uuid4().hex), evaluation_root=self.root,
            empty_summary=not before.present)
        after = read_published(self.source.maint_state_dir, self.source.user_id)
        frozen = read(target / 'snapshot.json')
        if ((before.version, before.text) != (after.version, after.text)
                or (before.present and frozen['summary_sha256'] != hashlib.sha256(before.text.encode()).hexdigest())):
            raise HTTPError('冻结期间摘要发生变化，请重新选择。', 409)
        return target

    def selected(self, mode, mid, text=None, *, need_history=False, frozen_current=None):
        if mode not in ('none', 'context', 'snapshot'):
            raise HTTPError('无效的记忆模式。')
        if text is not None and (mode != 'context' or mid is not None or not isinstance(text, str)):
            raise HTTPError('自定义全文仅用于 context，不能同时选择 memory id。')
        if mode == 'none' and (mid is not None or text is not None):
            raise HTTPError('none 模式不接受记忆内容。')
        selected = Published(False)
        snapshot = None
        if mode != 'none' and text is None:
            mid = mid or 'current'
            if mid not in self.memories:
                raise HTTPError('请选择已批准的记忆标识。')
            if mid == 'current':
                selected = read_published(self.source.maint_state_dir, self.source.user_id)
                if mode == 'snapshot' or need_history:
                    snapshot = frozen_current or self.freeze(selected)
                    selected = self.snapshot_published(snapshot)
            elif mid in self.snapshots:
                snapshot = self.snapshots[mid]
                evaluation._snapshot(snapshot, time.time())
                selected = self.snapshot_published(snapshot)
            else:
                if mode == 'snapshot':
                    raise HTTPError('该记忆没有对应历史快照。')
                selected = Published(True, version=self.memories[mid]['version'], text=self.memories[mid]['text'])
        elif text is not None:
            selected = Published(bool(text.strip()), text=text)
        if need_history and snapshot is None:
            snapshot = frozen_current or self.freeze()
        return selected, snapshot

    @staticmethod
    def snapshot_published(path):
        fields = read(path / 'snapshot.json')['published']
        return Published(**{**fields, 'text': (path / 'summary.txt').read_text('utf-8'),
                            'summary_path': str(path / 'summary.txt'), 'memory_path': None})

    def engine(self, config, session, *, source=None):
        if is_v1(session):
            return self.memory_v1.engine(session)
        from .sandbox_summary import SandboxSummary
        workspace_path = config.project_root / 'workspace.json'
        if workspace_path.exists():
            session = {**session, **read(workspace_path)}
        runtime = dataclasses.replace(config, maint_cli=self.source.maint_cli)
        options = self.runtime_summary(session['summary'], [])
        options.pop('engine', None)  # Workbench routing metadata is not a legacy engine option.
        return SandboxSummary(runtime, source_config=source or self.source,
                              initial=Published(**session['initial']),
                              options=options)

    def create_session(self, body):
        keys(body, {'model', 'memory', 'retrieval', 'summary', 'title', 'continue_from'})
        if body.get('continue_from') is not None:
            return self.continue_session(body)
        keys(body, {'model', 'memory', 'retrieval', 'summary', 'title'}, {'model', 'memory'})
        model = profile(body['model'])
        memory = body['memory']
        keys(memory, {'enabled', 'mode', 'id', 'text'}, {'mode'})
        if 'enabled' in memory and type(memory['enabled']) is not bool:
            raise HTTPError('memory.enabled 必须为布尔值。')
        retrieval = body.get('retrieval', {'enabled': False})
        keys(retrieval, {'enabled', 'profile'}, {'enabled'})
        if type(retrieval['enabled']) is not bool:
            raise HTTPError('retrieval.enabled 必须为布尔值。')
        rp = profile(retrieval.get('profile', self.retrieval_profile))
        summary = summary_options(body.get('summary'), self.pool)
        if is_v1({'summary': summary}) and memory['mode'] != 'none':
            raise HTTPError('新 v1 workspace 必须使用 memory.mode=none，从空记忆初始化。')
        title = body.get('title', '新会话')
        if not isinstance(title, str) or len(title) > 300:
            raise HTTPError('会话标题无效。')
        self.discover()
        # Freeze once for every chat workspace so later agent toggles keep the
        # same source, original chat#N IDs and any facts learned in this branch.
        initial, snapshot = self.selected(memory['mode'], memory.get('id'), memory.get('text'),
                                           need_history=True)
        sid = uuid.uuid4().hex
        folder = self.root / 'sessions' / sid
        folder.mkdir(mode=0o700)
        config = isolated(self.source, folder, model, rp)
        if snapshot is not None:
            evaluation._snapshot(snapshot, time.time())
            evaluation._backup(snapshot / 'history.sqlite', config.db_path, frozen=True)
        with closing(db.connect(config.db_path)) as conn:
            db.init_db(conn)
            conn.execute('DELETE FROM maint_queue')
            conn.commit()
        session = {'id': sid, 'workspace_id': sid, 'title': title, 'created_at': time.time(), 'model': model,
            'memory': memory, 'retrieval': {'enabled': retrieval['enabled'], 'profile': rp},
            'summary': summary, 'initial': dataclasses.asdict(initial), 'config': config_json(config),
            'history_snapshot_id': snapshot.name if snapshot else None,
            'memory_strategy': 'v1_empty_until_initialize' if is_v1({'summary': summary})
                               else 'selected_baseline_plus_new_turns'}
        write(folder / 'workspace.json', {'initial': session['initial'], 'summary': summary})
        if summary['enabled'] and not is_v1(session):
            engine = self.engine(config, session)
            status = engine.initialize()
            session['memory_strategy'] = status.get('seed_mode', session['memory_strategy'])
        write(folder / 'session.json', session)
        with self.guard:
            self.sessions[sid], self.locks[sid] = session, threading.Lock()
        return {'session': self.public_session(session)}

    def continue_session(self, body):
        previous = self.session(body['continue_from'])
        memory = copy.deepcopy(body.get('memory', previous['memory']))
        keys(memory, {'enabled', 'mode', 'id', 'text'}, {'mode'})
        if 'enabled' in memory and type(memory['enabled']) is not bool:
            raise HTTPError('memory.enabled 必须为布尔值。')
        source_fields = lambda m: {k: v for k, v in m.items() if k != 'enabled'}
        if source_fields(memory) != source_fields(previous['memory']):
            raise HTTPError('修改记忆来源需要新实验。', 409)
        requested_summary = body.get('summary', {})
        if (isinstance(requested_summary, dict) and 'engine' in requested_summary
                and requested_summary['engine'] != previous['summary'].get('engine', 'v2.6')):
            raise HTTPError('不能切换已有 workspace 的摘要引擎。', 409)
        summary = summary_options({**previous['summary'], **requested_summary}, self.pool)
        if {k: v for k, v in summary.items() if k != 'enabled'} != {
                k: v for k, v in previous['summary'].items() if k != 'enabled'}:
            raise HTTPError('修改摘要模型或累计预算需要新实验。', 409)
        retrieval = copy.deepcopy(body.get('retrieval', previous['retrieval']))
        keys(retrieval, {'enabled', 'profile'}, {'enabled'})
        if type(retrieval['enabled']) is not bool:
            raise HTTPError('retrieval.enabled 必须为布尔值。')
        retrieval['profile'] = profile(retrieval.get('profile', previous['retrieval']['profile']))
        if retrieval['enabled'] and not previous['history_snapshot_id']:
            raise HTTPError('该实验未冻结原文；开启历史检索需要新实验。', 409)
        model = profile(body.get('model', previous['model']))
        title = body.get('title', '新会话')
        if not isinstance(title, str) or len(title) > 300:
            raise HTTPError('会话标题无效。')
        lock = self.locks[previous['id']]
        if not lock.acquire(blocking=False):
            raise HTTPError('该实验正在回答或更新摘要。', 409)
        try:
            sid = uuid.uuid4().hex
            session = {**previous, 'id': sid, 'title': title, 'created_at': time.time(),
                       'model': model, 'memory': memory, 'retrieval': retrieval, 'summary': summary,
                       'continue_from': previous['id']}
            config = dataclasses.replace(restore_config(session['config']),
                main_chat_model=model['model'], chat_model=retrieval['profile']['model'],
                api_base_url=model['base_url'], api_key_env=model.get('api_key_env', 'WORKBENCH_UNUSED_KEY'))
            session['config'] = config_json(config)
            workspace_path = config.project_root / 'workspace.json'
            workspace = read(workspace_path)
            if summary['enabled'] and not workspace['summary']['enabled']:
                workspace['summary'] = summary
                write(workspace_path, workspace)
                if not is_v1(session):
                    self.engine(config, session).initialize()
            folder = self.root / 'sessions' / sid
            folder.mkdir(mode=0o700)
            write(folder / 'session.json', session)
            with self.guard:
                self.sessions[sid], self.locks[sid] = session, lock
            return {'session': self.public_session(session)}
        finally:
            lock.release()

    def published(self, session):
        if is_v1(session):
            return self.memory_v1.engine(session).published()
        config = restore_config(session['config'])
        workspace_path = config.project_root / 'workspace.json'
        stored = read(workspace_path) if workspace_path.exists() else session
        if stored['summary']['enabled']:
            return self.engine(config, session).published()
        return Published(**stored['initial'])

    def detail(self, sid):
        session = self.session(sid)
        config = restore_config(session['config'])
        with closing(db.connect(config.db_path)) as conn:
            rows = conn.execute('SELECT id,role,content,created_at FROM chat_log WHERE instance=? '
                'AND conversation_id=? ORDER BY id', (config.instance, sid)).fetchall()
        return {'session': self.public_session(session), 'messages': [dict(r) for r in rows],
                'memory': memory_json(self.published(session)),
                'runs': [copy.deepcopy(r) for r in self.runs.values() if r['session_id'] == sid]}

    def start_chat(self, sid, body):
        keys(body, {'request_id', 'message'}, {'request_id', 'message'})
        try:
            request_id = str(uuid.UUID(body['request_id']))
        except (ValueError, AttributeError, TypeError):
            raise HTTPError('request_id 必须为 UUID。') from None
        message = body['message']
        if not isinstance(message, str) or not message.strip():
            raise HTTPError('消息不能为空。')
        digest = hashlib.sha256(message.encode()).hexdigest()
        with self.guard:
            session = self.session(sid)
            for old in self.runs.values():
                if old['session_id'] == sid and old['request_id'] == request_id:
                    if old.get('kind', 'chat') != 'chat' or old.get('message_sha256') != digest:
                        raise HTTPError('该 request_id 已用于其他消息。', 409)
                    return {'run_id': old['id'], 'run': copy.deepcopy(old), 'reused': True}
            if len(self.runs) >= 10000:
                raise HTTPError('请求登记已满，仍可查看与回放已有请求。', 409)
            lock = self.locks[sid]
            if not lock.acquire(blocking=False):
                raise HTTPError('该会话正在回答或更新摘要。', 409)
            rid = uuid.uuid4().hex
            folder = self.root / 'runs' / rid
            folder.mkdir(mode=0o700)
            run = {'id': rid, 'run_id': rid, 'session_id': sid, 'request_id': request_id,
                'message_sha256': digest, 'status': 'running', 'created_at': time.time()}
            try:
                write(folder / 'run.json', run)
            except BaseException:
                lock.release()
                raise
            self.runs[rid], self.hubs[rid] = run, RunEvents(folder, run)
        threading.Thread(target=self._chat_worker, args=(session, run, message, lock), daemon=True).start()
        return {'run_id': rid, 'run': copy.deepcopy(run), 'reused': False}

    def _chat_worker(self, session, run, message, lock):
        folder = self.root / 'runs' / run['id']
        hub = self.hubs[run['id']]
        observer = Observer(hub, session_id=session['id'], request_id=run['request_id'], run_id=run['id'])
        config = restore_config(session['config'])
        profiles = [session['model'], session['retrieval']['profile'], session['summary']['profile']]
        hub.secrets = tuple(os.getenv(p['api_key_env']) for p in profiles
                            if p.get('api_key_env') and os.getenv(p['api_key_env']))
        def recorder(event):
            safe = redact_record(event, hub.secrets)
            # Preserve existing provider redaction metadata when already sanitized.
            if event.get('redactions'):
                safe['redactions'] = event['redactions'] + safe['redactions']
                safe.update(record_changed=True, replay_limited=True)
            append(folder / 'wire.jsonl', safe)
            observer.emit('provider_call', **safe)
        response = None
        try:
            with observe(observer), closing(db.connect(config.db_path)) as conn:
                observer.emit('request_accepted')
                client = create_client(config, session['model'], recorder=recorder, role='main')
                retrieval_client = (create_client(config, session['retrieval']['profile'], recorder=recorder,
                                                  role='retrieval') if session['retrieval']['enabled'] else None)
                inject = session['memory'].get('enabled', True)
                published = self.published(session) if inject else Published(False)
                response = chat.chat_agentic(config, conn, message, published=published,
                    instance=config.instance, conversation_id=session['id'], maintenance=None,
                    stream=True, client=client, retrieval_client=retrieval_client,
                    now_epoch=time.time(), allow_memory_tools=session['retrieval']['enabled'])
                write(folder / 'answer.json', response)
                observer.emit('answer_complete', result=response, answer=response['answer'])
                if session['summary']['enabled']:
                    observer.emit('maintenance_started')
                    try:
                        if is_v1(session):
                            self.memory_v1.after_chat(session, run, response, observer)
                        else:
                            engine = self.engine(config, session)
                            result = engine.update(conn, response['user_message_id'], response['assistant_message_id'],
                                               recorder=recorder)
                            write(folder / 'summary.json', result)
                            if result.get('status') in ('failed', 'error'):
                                observer.emit('maintenance_failed', error='摘要更新失败，保留已保存回答和原摘要。', result=result)
                            else:
                                observer.emit('maintenance_published', memory=memory_json(engine.published()), result=result)
                    except Exception:
                        observer.emit('maintenance_failed', error='摘要更新失败，保留已保存回答和原摘要。')
                with self.guard:
                    run.update(status='completed', finished_at=time.time())
        except Exception as exc:
            with self.guard:
                run.update(status='failed', finished_at=time.time(), error_type=type(exc).__name__,
                           error='请求失败，请查看脱敏的请求记录；未保存失败的助手回答。')
            observer.emit('request_failed', error_type=type(exc).__name__,
                          error='请求失败，请查看脱敏的请求记录；未保存失败的助手回答。')
        finally:
            try:
                write(folder / 'run.json', run)
                observer.emit('run_complete', status=run['status'], answer_saved=response is not None)
            finally:
                lock.release()

    def prepare_batch(self, body):
        keys(body, {'suite', 'summary', 'previous_id'}, {'suite'})
        summary = summary_options(body.get('summary'), self.pool)
        if is_v1({'summary': summary}):
            raise HTTPError('批测暂不支持 summary.engine=v1。')
        suite = copy.deepcopy(body['suite'])
        keys(suite, {'version', 'models', 'questions', 'contexts', 'repeats', 'concurrency',
                     'budgets', 'now_epoch', 'retrieval_profile', 'recent_seed', 'chat'},
                    {'models', 'questions', 'contexts', 'budgets'})
        suite.setdefault('version', 1)
        suite.setdefault('now_epoch', time.time())
        for key, limit in [('repeats', 100), ('concurrency', 16)]:
            value = suite.get(key, 1)
            if type(value) is not int or not 1 <= value <= limit:
                raise HTTPError('重复数或并发超出允许范围。')
        for group in ('models', 'questions', 'contexts'):
            if not isinstance(suite[group], list) or not 1 <= len(suite[group]) <= 100:
                raise HTTPError('矩阵列表不能为空或过大。')
        if len(suite['models']) * len(suite['questions']) * len(suite['contexts']) * suite.get('repeats', 1) > 10000:
            raise HTTPError('单批最多 10000 条测试。')
        suite['models'] = [profile(p) for p in suite['models']]
        pipeline = any(c.get('mode') == 'pipeline' for c in suite['contexts'] if isinstance(c, dict))
        if pipeline != ('retrieval_profile' in suite):
            raise HTTPError('仅 pipeline 实验必须设置独立 retrieval_profile。')
        if pipeline:
            suite['retrieval_profile'] = profile(suite['retrieval_profile'])
        previous = None
        if body.get('previous_id') is not None:
            previous = self.batch_path(body['previous_id']) / 'run'
            if not (previous / 'results.json').exists():
                raise HTTPError('所选回归批次尚无完整结果。')
        current = self.discover()
        fixed_current = None
        selected_initials = {}
        for context in suite['contexts']:
            keys(context, {'id', 'mode', 'memory_id', 'text', 'recent_seed', 'inject_memory'}, {'id', 'mode'})
            if 'inject_memory' in context and type(context['inject_memory']) is not bool:
                raise HTTPError('context.inject_memory 必须为布尔值。')
            identifier(context['id'])
            mode = context['mode']
            if mode not in ('none', 'context', 'snapshot', 'pipeline'):
                raise HTTPError('无效的实验材料模式。')
            mid, text = context.pop('memory_id', None), context.get('text')
            if mode == 'none':
                if mid is not None or text is not None or 'recent_seed' in context:
                    raise HTTPError('none 实验不接受记忆或近期材料。')
                selected_initials[context['id']] = Published(False)
                continue
            if mode in ('snapshot', 'pipeline') and text is not None:
                raise HTTPError('快照材料只能使用已批准的记忆标识。')
            if (mid or 'current') == 'current' and text is None and mode in ('snapshot', 'pipeline'):
                if fixed_current is None:
                    fixed_current = self.freeze()
            if mode == 'context' and (mid or 'current') == 'current' and text is None:
                initial, snapshot = current, None
            else:
                initial, snapshot = self.selected('snapshot' if mode in ('snapshot', 'pipeline') else 'context',
                                                   mid, text, frozen_current=fixed_current)
            selected_initials[context['id']] = initial
            if summary['enabled'] and mode == 'context' and mid in (None, 'current') and text is None:
                # The original evaluator has only an after-answer hook. Its
                # snapshot mode preserves source chat IDs before the answer;
                # static current memory with maintenance therefore uses that
                # same frozen history, with tools still disabled and no recents.
                if fixed_current is None:
                    fixed_current = self.freeze(current)
                snapshot = fixed_current
                mode = context['mode'] = 'snapshot'
                context.pop('text', None)
            if mode in ('snapshot', 'pipeline'):
                context['snapshot'] = str(snapshot)
            else:
                context['text'] = initial.text
        after = read_published(self.source.maint_state_dir, self.source.user_id)
        if (current.version, current.text) != (after.version, after.text):
            raise HTTPError('准备期间当前摘要发生变化，请重新规划。', 409)
        filename = self.root / 'suites' / (uuid.uuid4().hex + '.json')
        write(filename, suite)
        loaded = evaluation.load_suite(filename)
        plan = evaluation.plan_suite(loaded)
        if summary['enabled']:
            selected = summary['profile']
            if selected.get('auth') == 'codex_subscription':
                if selected['id'] not in plan['runtime_auth_unchecked']:
                    plan['runtime_auth_unchecked'].append(selected['id'])
            elif not os.environ.get(selected['api_key_env']):
                plan['missing_credentials'].append({'profile': selected['id'],
                    'api_key_env': selected['api_key_env'], 'base_url': selected['base_url'], 'role': 'summary'})
                plan['valid'] = False
        plan['summary'] = {**summary, 'additional_to_chat_budget': True,
            'budget_scope': 'cumulative_per_trial_workspace',
            'batch_request_ceiling': summary['max_requests'] * len(plan['cases']) if summary['enabled'] else 0,
            'memory_strategy': 'matching_structured_clone_or_fixed_baseline_plus_new_turns',
            'seed_mode': 'selected_summary; full source clone only when exact summary matches'}
        plan['context_modes'] = [{'id': c['id'], 'mode': c['mode'],
                                  'inject_memory': c.get('inject_memory', True)}
                                 for c in loaded['suite']['contexts']]
        plan['batch_request_ceiling'] = plan['per_trial_request_ceiling'] * len(plan['cases'])
        plan['warnings'] = [f"主模型与检索合计：每条最多 {plan['per_trial_request_ceiling']} 次请求，"
            f"全批 {len(plan['cases'])} 条合计上限 {plan['batch_request_ceiling']} 次。"]
        if summary['enabled']:
            plan['warnings'].append(f"摘要维护另计：每个私有实验空间累计最多新增 {summary['max_requests']} 次请求，"
                f"全批额外上限 {plan['summary']['batch_request_ceiling']} 次；输出固定 64000 tokens。")
        return filename, loaded, plan, summary, previous, selected_initials

    def batch_path(self, bid):
        identifier(bid)
        if bid not in self.batches:
            raise HTTPError('批次不存在。', 404)
        return self.root / 'batches' / bid

    def start_batch(self, body):
        filename, loaded, plan, summary, previous, initials = self.prepare_batch(body)
        if not plan['valid']:
            raise HTTPError('环境变量凭据缺失，请先查看执行计划。')
        bid = uuid.uuid4().hex
        folder = self.root / 'batches' / bid
        folder.mkdir(mode=0o700)
        sources, seeds = {}, {}
        source_before = read_published(self.source.maint_state_dir, self.source.user_id)
        for context in body['suite']['contexts']:
            if (context['mode'] != 'none' and context.get('memory_id', 'current') == 'current'
                    and 'text' not in context):
                selected = initials[context['id']]
                if (selected.version, selected.text) != (source_before.version, source_before.text):
                    raise HTTPError('摘要在冻结后发生变化，请重新开始实验。', 409)
        # Clone matching structured current memory ONCE before scheduling models.
        if summary['enabled']:
            for cid, initial in initials.items():
                identity = (initial.present, initial.user_id, initial.version, initial.text)
                if identity in seeds:
                    sources[cid] = seeds[identity]
                    continue
                seed = self.root / 'batch-inputs' / bid / cid
                seed.mkdir(parents=True)
                cfg = isolated(self.source, seed, loaded['suite']['models'][0],
                               loaded['suite'].get('retrieval_profile', self.retrieval_profile))
                record = {'initial': dataclasses.asdict(initial), 'summary': summary}
                self.engine(cfg, record).initialize()
                sources[cid] = seeds[identity] = cfg
        source_after = read_published(self.source.maint_state_dir, self.source.user_id)
        if (source_before.version, source_before.text) != (source_after.version, source_after.text):
            raise HTTPError('冻结摘要分支时来源发生变化，请重新开始实验。', 409)
        batch = {'id': bid, 'status': 'running', 'created_at': time.time(), 'plan': plan,
                 'summary': summary, 'previous_id': body.get('previous_id')}
        write(folder / 'batch.json', batch)
        with self.guard:
            self.batches[bid] = batch
        def after_trial(config, conn, response, case):
            initial = initials[case['context']['id']]
            config.user_id = sources[case['context']['id']].user_id
            record = {'initial': dataclasses.asdict(initial), 'summary': summary}
            engine = self.engine(config, record, source=sources[case['context']['id']])
            engine.initialize()
            def recorder(event):
                append(config.project_root / 'wire.jsonl', event)
            return engine.update(conn, response['user_message_id'], response['assistant_message_id'], recorder=recorder)
        def work():
            try:
                options = {'out': folder / 'run', 'evaluation_root': self.root, 'previous': previous}
                if summary['enabled']:
                    options['after_trial'] = after_trial
                evaluation.run_suite(filename, **options)
                manifest = read(folder / 'run' / 'manifest.json')
                batch['status'] = 'completed' if manifest['status'] == 'complete' else manifest['status']
            except Exception as exc:
                batch.update(status='failed', error={'type': type(exc).__name__,
                    'message': '批次执行失败；请查看已保存结果和脱敏请求记录。'})
            finally:
                batch['finished_at'] = time.time()
                write(folder / 'batch.json', batch)
        threading.Thread(target=work, daemon=True).start()
        return {'batch': copy.deepcopy(batch)}

    def resume_batch(self, bid, body):
        keys(body, {'concurrency'}, {'concurrency'})
        concurrency = body['concurrency']
        if type(concurrency) is not int or not 1 <= concurrency <= 16:
            raise HTTPError('并发必须为 1–16。')
        folder = self.batch_path(bid)
        run = folder / 'run'
        loaded = evaluation.load_suite(run / 'inputs' / 'materialized-suite.json')
        suite = loaded['suite']
        batch = self.batches[bid]
        summary = self.runtime_summary(batch['summary'], [])
        if is_v1({'summary': summary}):
            raise HTTPError('批测暂不支持 summary.engine=v1。')
        initials, sources = {}, {}
        if summary['enabled']:
            for context in suite['contexts']:
                cid = context['id']
                seed = self.root / 'batch-inputs' / bid / cid
                metadata = read(seed / 'state' / 'sandbox-summary.json')
                present, user_id, version, expected_sha = metadata['initial_identity']
                text = context['text']
                if hashlib.sha256(text.encode()).hexdigest() != expected_sha:
                    raise HTTPError('续跑材料与原摘要起点不同，已停止恢复。', 409)
                initials[cid] = Published(present, version=version, user_id=user_id, text=text)
                sources[cid] = isolated(self.source, seed, suite['models'][0],
                    suite.get('retrieval_profile', self.retrieval_profile))
                sources[cid].user_id = metadata['user_id']
        with self.guard:
            if batch['status'] == 'running':
                raise HTTPError('批次仍在运行，不能重复续跑。', 409)
            batch.update(status='running', runtime_concurrency=concurrency,
                         resumed_at=time.time())
            batch.pop('finished_at', None)
            batch.pop('error', None)
            write(folder / 'batch.json', batch)

        def after_trial(config, conn, response, case):
            cid = case['context']['id']
            config.user_id = sources[cid].user_id
            record = {'initial': dataclasses.asdict(initials[cid]), 'summary': summary}
            engine = self.engine(config, record, source=sources[cid])
            engine.initialize()
            spent = case.get('_resume_summary_usage', {})
            if any(spent.values()):
                ledger_path = config.maint_state_dir / 'ledger' / 'usage.json'
                ledger = read(ledger_path)
                for key in ('requests', 'input_tokens', 'output_tokens', 'errors'):
                    ledger[key] = ledger.get(key, 0) + spent.get(key, 0)
                write(ledger_path, ledger)
            def recorder(event):
                append(config.project_root / 'wire.jsonl', event)
            return engine.update(conn, response['user_message_id'], response['assistant_message_id'],
                                 recorder=recorder)

        def work():
            try:
                evaluation.resume_run(run, concurrency=concurrency,
                    after_trial=after_trial if summary['enabled'] else None)
                batch['status'] = ('completed' if read(run / 'manifest.json')['status'] == 'complete'
                                   else 'interrupted')
            except Exception as exc:
                batch.update(status='failed', error={'type': type(exc).__name__,
                    'message': '续跑失败，已完成结果保留；请查看保存的请求和恢复记录。'})
            finally:
                batch['finished_at'] = time.time()
                write(folder / 'batch.json', batch)
        threading.Thread(target=work, daemon=True).start()
        return {'batch': copy.deepcopy(batch)}

    def batch_detail(self, bid):
        folder = self.batch_path(bid)
        records = sorted(jsonlines(folder / 'run' / 'results.jsonl'), key=lambda r: r['index'])
        manifest = folder / 'run' / 'manifest.json'
        batch = copy.deepcopy(self.batches[bid])
        migrations = batch.setdefault('protocol_migrations', [])
        batch['summary'] = self.runtime_summary(batch['summary'], migrations)
        suite_file = folder / 'run' / 'inputs' / 'suite.json'
        if suite_file.exists():
            suite = read(suite_file)
            batch['runtime_profiles'] = {
                'main': [runtime_profile(p, migrations, 'main') for p in suite['models']],
                'summary': batch['summary']['profile']}
            if 'retrieval_profile' in suite:
                batch['runtime_profiles']['retrieval'] = runtime_profile(suite['retrieval_profile'], migrations, 'retrieval')
        return {'batch': batch, 'manifest': read(manifest) if manifest.exists() else None, 'results': records}


def make_handler(workbench):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, value, content_type='application/json; charset=utf-8'):
            body = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def local(self):
            host = self.headers.get('Host', '')
            allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}',
                       f'[::1]:{self.server.server_port}'}
            origin = self.headers.get('Origin')
            if (host not in allowed or origin is not None and origin != 'http://' + host
                    or self.headers.get('Sec-Fetch-Site') == 'cross-site'):
                raise HTTPError('仅允许本机页面访问。', 403)

        def body(self):
            if self.headers.get('Transfer-Encoding'):
                raise HTTPError('不支持分块请求体。')
            if self.headers.get_content_type() != 'application/json':
                raise HTTPError('请发送 JSON。', 415)
            try:
                length = int(self.headers.get('Content-Length', '0'))
            except ValueError:
                raise HTTPError() from None
            if not 0 < length <= MAX_BODY:
                raise HTTPError('请求体为空或超过 32MB。', 413)
            def pairs(items):
                result = {}
                for key, value in items:
                    if key in result:
                        raise ValueError('duplicate JSON field')
                    result[key] = value
                return result
            try:
                return json.loads(self.rfile.read(length).decode('utf-8'), object_pairs_hook=pairs,
                                  parse_constant=lambda v: (_ for _ in ()).throw(ValueError()))
            except (ValueError, UnicodeDecodeError):
                raise HTTPError('JSON 格式无效。') from None

        def do_GET(self):
            self.dispatch(False)

        def do_POST(self):
            self.dispatch(True)

        def do_DELETE(self):
            self.dispatch(False)

        def dispatch(self, post):
            try:
                self.local()
                path = urlsplit(self.path).path
                if self.command == 'DELETE':
                    parts = path.strip('/').split('/')
                    if len(parts) == 3 and parts[:2] == ['workbench', 'models']:
                        return self.send(200, workbench.delete_model(parts[2]))
                    if len(parts) == 3 and parts[:2] == ['workbench', 'sessions']:
                        return self.send(200, workbench.delete_session(parts[2]))
                    raise HTTPError('接口不存在。', 404)
                if not post and path in ('/', '/workbench', '/workbench/', '/workbench.html', '/workbench.js', '/workbench.css'):
                    name = 'workbench.html' if path in ('/', '/workbench', '/workbench/') else path[1:]
                    asset = Path(__file__).with_name(name)
                    if not asset.is_file():
                        raise HTTPError('页面资源尚未就绪。', 503)
                    kind = {'html': 'text/html', 'js': 'text/javascript', 'css': 'text/css'}[asset.suffix[1:]]
                    return self.send(200, asset.read_bytes(), kind + '; charset=utf-8')
                parts = path.strip('/').split('/')
                body = self.body() if post else None
                if parts == ['workbench', 'models']:
                    return self.send(200, workbench.save_model(body) if post else workbench.models())
                if parts == ['workbench', 'models', 'test'] and post:
                    return self.send(200, workbench.test_model(body))
                if parts == ['workbench', 'models', 'discover'] and post:
                    return self.send(200, workbench.discover_models(body))
                if parts == ['workbench', 'bootstrap'] and not post:
                    return self.send(200, workbench.bootstrap())
                if parts == ['workbench', 'sessions'] and post:
                    return self.send(201, workbench.create_session(body))
                if parts == ['workbench', 'batches', 'plan'] and post:
                    return self.send(200, workbench.prepare_batch(body)[2])
                if parts == ['workbench', 'batches'] and post:
                    return self.send(202, workbench.start_batch(body))
                if (len(parts) == 4 and parts[:2] == ['workbench', 'batches']
                        and parts[3] == 'resume' and post):
                    return self.send(202, workbench.resume_batch(parts[2], body))
                if len(parts) >= 3 and parts[:2] == ['workbench', 'sessions']:
                    sid = identifier(parts[2])
                    if len(parts) == 4 and parts[3] == 'rename' and post:
                        return self.send(200, workbench.rename_session(sid, body))
                    if len(parts) == 3 and not post:
                        return self.send(200, workbench.detail(sid))
                    if len(parts) == 4 and parts[3] == 'memory' and not post:
                        return self.send(200, memory_json(workbench.published(workbench.session(sid))))
                    if len(parts) >= 5 and parts[3:5] == ['memory', 'v1']:
                        if len(parts) == 5 and not post:
                            return self.send(200, workbench.memory_v1.get(sid))
                        if len(parts) == 6 and parts[5] == 'versions' and not post:
                            return self.send(200, workbench.memory_v1.get(sid, versions=True))
                        if len(parts) == 7 and parts[5] == 'versions' and not post:
                            return self.send(200, workbench.memory_v1.get(sid, parts[6]))
                        if len(parts) == 6 and parts[5] == 'jobs' and post:
                            return self.send(202, workbench.memory_v1.start(sid, body))
                    if len(parts) == 4 and parts[3] == 'chat' and post:
                        return self.send(202, workbench.start_chat(sid, body))
                if len(parts) >= 3 and parts[:2] == ['workbench', 'runs'] and not post:
                    rid = identifier(parts[2])
                    if rid not in workbench.runs:
                        raise HTTPError('请求不存在。', 404)
                    folder = workbench.root / 'runs' / rid
                    if len(parts) == 4 and parts[3] == 'events':
                        return self.sse(rid)
                    if len(parts) == 4 and parts[3] in ('wire', 'artifacts'):
                        return self.send(200, {'run': workbench.runs[rid],
                                              'wire_events': jsonlines(folder / 'wire.jsonl')})
                    if len(parts) == 3:
                        return self.send(200, {'run': workbench.runs[rid]})
                if len(parts) >= 3 and parts[:2] == ['workbench', 'batches'] and not post:
                    bid = identifier(parts[2])
                    folder = workbench.batch_path(bid) / 'run'
                    if len(parts) == 3:
                        return self.send(200, workbench.batch_detail(bid))
                    if len(parts) == 4 and parts[3] == 'report':
                        report = folder / 'comparison.html'
                        if not report.exists():
                            raise HTTPError('报告尚未生成。', 404)
                        return self.send(200, report.read_bytes(), 'text/html; charset=utf-8')
                    if len(parts) == 5 and parts[3] == 'results':
                        cid = identifier(parts[4])
                        result = folder / 'trials' / cid / 'result.json'
                        if not result.is_file():
                            raise HTTPError('结果不存在。', 404)
                        record = read(result)
                        return self.send(200, {'result': record,
                            'wire_events': jsonlines(result.parent / 'wire.jsonl'), 'summary': record.get('summary')})
                raise HTTPError('接口不存在。', 404)
            except HTTPError as exc:
                self.send(exc.status, {'error': str(exc)})
            except (ValueError, KeyError, TypeError, AttributeError):
                self.send(400, {'error': '参数无效；请核对选择和配置。'})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:
                self.send(500, {'error': '本地操作失败；请检查工作台状态。'})

        def sse(self, rid):
            query = parse_qs(urlsplit(self.path).query)
            cursor = self.headers.get('Last-Event-ID') or query.get('after', ['0'])[0]
            try:
                after = int(cursor)
                if after < 0:
                    raise ValueError()
            except ValueError:
                raise HTTPError('无效的事件序号。') from None
            hub = workbench.hubs[rid]
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
            self.send_header('Cache-Control', 'no-cache, no-store')
            self.send_header('X-Accel-Buffering', 'no')
            self.send_header('Connection', 'close')
            self.end_headers()
            self.close_connection = True
            if after > hub.seq:
                run = workbench.runs[rid]
                reset = {'phase': 'replay_gap', 'seq': 0, 'run_id': rid, 'root_run_id': rid,
                         'session_id': run['session_id'], 'request_id': run['request_id'],
                         'data': {'reason': 'cursor_ahead', 'reset_after': 0}}
                self.wfile.write(('event: reset\ndata: ' + json.dumps(reset) + '\n\n').encode())
                after = 0
            # Each subscriber owns its disk cursor, never the session lock or DB.
            with (hub.folder / 'events.jsonl').open('r', encoding='utf-8') as handle:
                while True:
                    position = handle.tell()
                    line = handle.readline()
                    if line and line.endswith('\n'):
                        event = json.loads(line)
                        if event['seq'] > after:
                            self.wfile.write(f'id: {event["seq"]}\ndata: {line.strip()}\n\n'.encode())
                            self.wfile.flush()
                            after = event['seq']
                        if event['phase'] == 'run_complete':
                            return
                    else:
                        handle.seek(position)
                        with hub.condition:
                            hub.condition.wait(timeout=1)
                        self.wfile.write(b': heartbeat\n\n')
                        self.wfile.flush()
    return Handler


def serve_workbench(source_config=None, *, host='127.0.0.1', port=8794, root=None):
    """Return an unstarted server. ``root`` is the complete private workbench root."""
    if host not in {'127.0.0.1', 'localhost', '::1'}:
        raise ValueError('loopback only')
    source = source_config if source_config is not None else load_config()
    workbench = Workbench(source, root or source.project_root / 'evaluations' / 'workbench')
    server_class = ThreadingHTTPServer
    if host == '::1':
        class IPv6Server(ThreadingHTTPServer):
            address_family = socket.AF_INET6
        server_class = IPv6Server
    server = server_class((host, port), make_handler(workbench))
    server.workbench = workbench
    return server
