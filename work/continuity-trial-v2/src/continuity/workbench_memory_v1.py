"""Workbench v1 jobs: frozen workspace inputs, durable runs, no automatic retries."""
from __future__ import annotations

import copy
import dataclasses
from datetime import datetime
from contextlib import closing
import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid

from . import db, evaluation
from .events import Observer, observe
from .providers import redact_record
from .retrieval import TrialRetrieval, now_shanghai


def is_v1(session):
    return session['summary'].get('engine', 'v2.6') == 'v1'


def aware_now(value, ceiling):
    from .workbench import HTTPError
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.timestamp() > ceiling:
            raise ValueError()
        return parsed.isoformat()
    except (ValueError, TypeError, OverflowError):
        raise HTTPError('now 必须为不晚于服务器当前时间的带时区 ISO 时间。') from None


class WorkbenchMemoryV1:
    def __init__(self, workbench):
        self.workbench = workbench

    def config(self, session):
        from .workbench import restore_config
        config = restore_config(session['config'])
        options = session['summary']
        return dataclasses.replace(config,
            usage_file=config.project_root / 'memory-v1-usage.json',
            max_requests=options['max_requests'], trial_max_requests=options['max_requests'],
            maint_total_budget=options['max_requests'],
            maint_max_output_tokens=options['max_output_tokens'],
            maint_model=options['profile']['model'], chat_model=options['profile']['model'])

    def engine(self, session):
        from .memory_v1 import MemoryV1
        from .workbench import HTTPError, read
        if not is_v1(session):
            raise HTTPError('该 workspace 不使用 v1 摘要引擎。', 409)
        config = self.config(session)
        workspace = read(config.project_root / 'workspace.json')
        if not is_v1(workspace):
            raise HTTPError('不能切换已有 workspace 的摘要引擎。', 409)
        return MemoryV1(config.maint_state_dir / 'memory-v1', config)

    def get(self, sid, number=None, *, versions=False):
        from .workbench import HTTPError
        engine = self.engine(self.workbench.session(sid))
        if versions:
            return {'versions': engine.versions()}
        if number is None:
            return engine.current()
        try:
            if not number.isdecimal():
                raise ValueError()
            return engine.version(int(number))
        except (ValueError, KeyError):
            raise HTTPError('v1 版本不存在。', 404) from None

    def validate(self, body):
        from .workbench import HTTPError, keys
        keys(body, {'request_id', 'operation', 'expected_version', 'now', 'batches', 'source_ids'},
             {'request_id', 'operation', 'expected_version'})
        request = copy.deepcopy(body)
        try:
            request['request_id'] = str(uuid.UUID(body['request_id']))
        except (ValueError, TypeError, AttributeError):
            raise HTTPError('request_id 必须为 UUID。') from None
        operation = body['operation']
        if operation not in ('initialize', 'append', 'refresh_stable'):
            raise HTTPError('不支持的 v1 operation。')
        if type(body['expected_version']) is not int or body['expected_version'] < 0:
            raise HTTPError('expected_version 必须为非负整数。')
        ceiling = time.time()
        if 'now' in body:
            request['now'] = aware_now(body['now'], ceiling)

        def source_ids(value):
            if (not isinstance(value, list) or any(not isinstance(s, str) or not s.strip() for s in value)
                    or len(set(value)) != len(value)):
                raise HTTPError('source_ids 必须为不重复的来源 ID 数组。')

        if operation == 'initialize':
            if 'source_ids' in body or not isinstance(body.get('batches'), list) or not body['batches']:
                raise HTTPError('initialize 必须提供非空 batches，不能提供顶层 source_ids。')
            previous = None
            for batch in request['batches']:
                keys(batch, {'now', 'source_ids'}, {'now', 'source_ids'})
                batch['now'] = aware_now(batch['now'], ceiling)
                source_ids(batch['source_ids'])
                if not batch['source_ids']:
                    raise HTTPError('初始化批次 source_ids 不能为空。')
                if previous is not None and datetime.fromisoformat(batch['now']) <= datetime.fromisoformat(previous):
                    raise HTTPError('初始化批次 now 必须严格递增。')
                previous = batch['now']
            if 'now' in request and datetime.fromisoformat(previous) > datetime.fromisoformat(request['now']):
                raise HTTPError('初始化批次不能晚于任务 now。')
        else:
            if 'batches' in body:
                raise HTTPError('仅 initialize 接受 batches。')
            if operation == 'append' and not body.get('source_ids'):
                raise HTTPError('append 必须提供非空 source_ids。')
            if operation == 'refresh_stable' and 'source_ids' in body:
                raise HTTPError('refresh_stable 自主检索，不接受 source_ids。')
            if 'source_ids' in body:
                source_ids(body['source_ids'])
        # Hash caller input before filling server now, so reconnects without now reuse the run.
        digest = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False,
            separators=(',', ':')).encode()).hexdigest()
        request.setdefault('now', now_shanghai())
        return request, digest

    @staticmethod
    def validate_sources(conn, config, request):
        from .workbench import HTTPError
        groups = request.get('batches') or [{'now': request['now'], 'source_ids': request.get('source_ids', [])}]
        for group in groups:
            view = TrialRetrieval(conn, config.instance, datetime.fromisoformat(group['now']).timestamp())
            for sid in group['source_ids']:
                source = view.by_sid.get(sid)
                if source is None or source['role'] != 'user':
                    raise HTTPError('来源必须是该 workspace 中早于 now 的有效用户原文 ID。')

    def start(self, sid, body):
        from .workbench import HTTPError, RunEvents, write
        wb = self.workbench
        request, digest = self.validate(body)
        with wb.guard:
            session = wb.session(sid)
            if not is_v1(session):
                raise HTTPError('该 workspace 不使用 v1 摘要引擎。', 409)
            workspace_id = session.get('workspace_id', sid)
            for old in wb.runs.values():
                if old['request_id'] != request['request_id']:
                    continue
                if old['session_id'] == sid and old.get('kind', 'chat') != 'memory_v1':
                    raise HTTPError('该 request_id 已用于聊天。', 409)
                if old.get('kind') == 'memory_v1' and old.get('workspace_id') == workspace_id:
                    if old['request_sha256'] != digest:
                        raise HTTPError('该 request_id 已用于不同的摘要任务。', 409)
                    return {'run_id': old['id'], 'run': copy.deepcopy(old), 'reused': True}
            if len(wb.runs) >= 10000:
                raise HTTPError('请求登记已满。', 409)
            lock = wb.locks[sid]
            if not lock.acquire(blocking=False):
                raise HTTPError('该 workspace 正在回答或更新摘要。', 409)
            try:
                engine = self.engine(session)
                current = engine.current()
                if current['version'] != request['expected_version']:
                    raise HTTPError('expected_version 与当前 v1 版本冲突。', 409)
                if request['operation'] == 'initialize' and current['version'] != 0:
                    raise HTTPError('initialize 只接受空版本 0。', 409)
                if current['now_max'] and datetime.fromisoformat(request['now']) < datetime.fromisoformat(current['now_max']):
                    raise HTTPError('任务 now 不能早于当前版本 now_max。')
                config = self.config(session)
                # Check IDs before accepting a task; the lock keeps the workspace input fixed.
                with closing(db.connect(config.db_path)) as conn:
                    self.validate_sources(conn, config, request)
                rid = uuid.uuid4().hex
                folder = wb.root / 'runs' / rid
                folder.mkdir(mode=0o700)
                evaluation._backup(config.db_path, folder / 'sources.sqlite')
                os.chmod(folder / 'sources.sqlite', 0o600)
                run = {'id': rid, 'run_id': rid, 'kind': 'memory_v1', 'session_id': sid,
                    'workspace_id': workspace_id, 'request_id': request['request_id'],
                    'operation_id': f'{workspace_id}:{request["request_id"]}',
                    'operation': request['operation'], 'request_sha256': digest,
                    'status': 'running', 'created_at': time.time()}
                write(folder / 'request.json', request)
                write(folder / 'run.json', run)
                wb.runs[rid], wb.hubs[rid] = run, RunEvents(folder, run)
                threading.Thread(target=self.worker, args=(session, run, request, lock), daemon=True).start()
            except BaseException:
                lock.release()
                raise
        return {'run_id': rid, 'run': copy.deepcopy(run), 'reused': False}

    @staticmethod
    def safe_error(exc, secrets=()):
        # ProviderError messages are local diagnostics; arbitrary runtime errors may echo input/keys.
        from .memory_v1 import MemoryV1Error
        from .deepseek import BudgetExceeded, MissingCredential
        from .providers import ProviderError
        message = (str(exc)[:1200] if isinstance(exc, (MemoryV1Error, BudgetExceeded,
                    MissingCredential, ProviderError)) else
                   '摘要执行失败；已保留旧版，请查看脱敏 wire 和版本后显式重提。')
        safe = redact_record({'error': message, 'error_type': type(exc).__name__}, secrets)
        return {key: safe[key] for key in ('error', 'error_type')}

    def context(self, session, run):
        from .workbench import append
        folder = self.workbench.root / 'runs' / run['id']
        hub = self.workbench.hubs[run['id']]
        selected = session['summary']['profile']
        secret = os.getenv(selected.get('api_key_env', ''))
        if secret:
            hub.secrets = tuple(set(hub.secrets) | {secret})
        observer = Observer(hub, session_id=session['id'], request_id=run['request_id'],
                            run_id=run['id'], channel='summary')

        def recorder(event):
            safe = redact_record(event, hub.secrets)
            if event.get('redactions'):
                safe['redactions'] = event['redactions'] + safe['redactions']
                safe.update(record_changed=True, replay_limited=True)
            append(folder / 'wire.jsonl', safe)
            observer.emit('provider_call', **safe)
        return observer, recorder

    def execute(self, session, run, request, conn, observer, recorder):
        from .workbench import create_client
        from .deepseek import BudgetExceeded
        config = self.config(session)
        if session['summary']['max_requests'] == 0:
            raise BudgetExceeded('summary.max_requests=0；摘要请求预算为零。')
        selected = copy.deepcopy(session['summary']['profile'])
        parameters = selected.setdefault('parameters', {})
        parameters['max_output_tokens'] = min(parameters.get('max_output_tokens', 64000),
                                               session['summary']['max_output_tokens'])
        client = create_client(config, selected, recorder=recorder, role='summary')
        return self.engine(session).execute(request['operation'], conn=conn, client=client,
            now=request['now'], operation_id=run['operation_id'],
            expected_version=request['expected_version'], batches=request.get('batches'),
            source_ids=request.get('source_ids'), emit=observer.emit)

    @staticmethod
    def recording_warning(run, stage, exc):
        run.setdefault('summary_warnings', []).append({'stage': stage,
            'error_type': type(exc).__name__,
            'message': '摘要结果已确定，但产物或事件记录失败；请以 run 中结果和实际版本为准。'})

    def record_summary(self, run, result, observer):
        """Recording errors cannot change the already determined engine outcome."""
        from .workbench import write
        folder = self.workbench.root / 'runs' / run['id']
        try:
            write(folder / 'summary.json', result)
        except Exception as exc:
            self.recording_warning(run, 'summary_artifact', exc)
        try:
            observer.emit('maintenance_' + result['status'], result=result,
                          error=result.get('error'), error_type=result.get('error_type'))
        except Exception as exc:
            self.recording_warning(run, 'summary_observer', exc)

    def worker(self, session, run, request, lock):
        from .workbench import write
        folder = self.workbench.root / 'runs' / run['id']
        observer = Observer(self.workbench.hubs[run['id']], session_id=session['id'],
            request_id=run['request_id'], run_id=run['id'], channel='summary')
        result = None
        try:
            observer, recorder = self.context(session, run)
            observer.emit('request_accepted', operation=request['operation'])
            with observe(observer), closing(sqlite3.connect(
                    (folder / 'sources.sqlite').as_uri() + '?mode=ro', uri=True)) as conn:
                conn.row_factory = sqlite3.Row
                result = self.execute(session, run, request, conn, observer, recorder)
        except Exception as exc:
            if result is None:
                error = self.safe_error(exc, self.workbench.hubs[run['id']].secrets)
                run.update(status='failed', **error)
            else:
                self.recording_warning(run, 'source_cleanup', exc)
        finally:
            try:
                if result is not None:
                    run.update(status='completed', result=result)
                self.record_summary(run, result if result is not None else {
                    'status': 'failed', 'error': run['error'], 'error_type': run['error_type']}, observer)
                run['finished_at'] = time.time()
                write(folder / 'run.json', run)
                try:
                    observer.emit('run_complete', status=run['status'], result=run.get('result'),
                                  error=run.get('error'), error_type=run.get('error_type'))
                except Exception as exc:
                    self.recording_warning(run, 'run_complete_observer', exc)
                    write(folder / 'run.json', run)
            finally:
                lock.release()

    def after_chat(self, session, run, response, observer):
        from .workbench import write
        folder = self.workbench.root / 'runs' / run['id']
        result = None
        try:
            summary_observer, summary_recorder = self.context(session, run)
            uid = response['user_message_id']
            source_id = uid if isinstance(uid, str) and uid.startswith('chat#') else f'chat#{uid}'
            request = {'operation': 'append', 'expected_version': self.engine(session).current()['version'],
                'now': now_shanghai(), 'source_ids': [source_id]}
            operation = {**run, 'operation_id': f'chat:{run["id"]}:append'}
            write(folder / 'summary-request.json', {**request, 'operation_id': operation['operation_id']})
            evaluation._backup(self.config(session).db_path, folder / 'summary-sources.sqlite')
            os.chmod(folder / 'summary-sources.sqlite', 0o600)
            with closing(sqlite3.connect((folder / 'summary-sources.sqlite').as_uri() + '?mode=ro', uri=True)) as conn:
                conn.row_factory = sqlite3.Row
                result = self.execute(session, operation, request, conn, summary_observer, summary_recorder)
        except Exception as exc:
            if result is None:
                result = {'status': 'failed', **self.safe_error(exc, self.workbench.hubs[run['id']].secrets)}
            else:
                self.recording_warning(run, 'source_cleanup', exc)
        run['summary'] = result
        self.record_summary(run, result, observer)

    def recover(self, run):
        """Inspect committed versions only. Never create a client during boot."""
        from .workbench import read
        try:
            sid = run['session_id']
            stored = self.workbench.sessions.get(sid) or read(
                self.workbench.root / 'sessions' / sid / 'session.json')
            engine = self.engine(stored)
            for version in engine.versions():
                memory = engine.version(version) if type(version) is int else version
                if memory.get('operation_id') == run['operation_id']:
                    run.update(status='completed', result={'status': 'published',
                        'version': memory['version'], 'memory': memory}, recovered=True)
                    return
        except Exception:
            pass
        run.update(status='interrupted', error_type='InterruptedMemoryJob',
            error='服务重启，未自动重发摘要请求；请查看 v1 版本确认是否已提交，再以新 request_id 显式重提。')

    def recover_chat(self, run, folder):
        from .workbench import read
        request = read(folder / 'summary-request.json')
        summary_run = {**run, 'operation_id': request['operation_id']}
        self.recover(summary_run)
        if summary_run['status'] == 'completed':
            run.update(status='completed', summary=summary_run['result'], recovered=True,
                       answer_saved=(folder / 'answer.json').is_file())
        else:
            run.update(error=summary_run['error'], error_type=summary_run['error_type'],
                       summary={'status': 'interrupted', 'error': summary_run['error']})
