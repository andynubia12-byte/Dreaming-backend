"""Session metadata changes; workspace data and run artifacts are retained."""
from __future__ import annotations

import copy
from contextlib import contextmanager
import time
import unicodedata


@contextmanager
def _locked_session(workbench, sid, *, allow_deleted=False):
    from .workbench import HTTPError, identifier, read

    identifier(sid)
    with workbench.guard:
        session = workbench.sessions.get(sid)
        if session is None or session.get('deleted_at') is not None:
            path = workbench.root / 'sessions' / sid / 'session.json'
            if allow_deleted and path.is_file():
                stored = read(path)
                if stored.get('id') == sid and stored.get('deleted_at') is not None:
                    yield None
                    return
            raise HTTPError('会话不存在。', 404)
        lock = workbench.locks[sid]
        if not lock.acquire(blocking=False):
            raise HTTPError('该实验正在回答或更新摘要。', 409)
        try:
            workspace = session.get('workspace_id', sid)
            for run in workbench.runs.values():
                other_id = run['session_id']
                other = workbench.sessions.get(other_id, {})
                if run['status'] in ('running', 'accepted') and (
                        other_id == sid or other.get('workspace_id', other_id) == workspace
                        or workbench.locks.get(other_id) is lock):
                    raise HTTPError('该实验正在回答或更新摘要。', 409)
            yield session
        finally:
            lock.release()


def rename_session(workbench, sid, body):
    from .workbench import HTTPError, keys, write

    keys(body, {'title'}, {'title'})
    title = body['title']
    if (not isinstance(title, str)
            or any(unicodedata.category(char) in ('Cc', 'Cs', 'Zl', 'Zp') for char in title)):
        raise HTTPError('会话标题无效。')
    title = title.strip()
    if not title or len(title) > 300:
        raise HTTPError('会话标题必须为 1–300 个字符。')
    with _locked_session(workbench, sid) as session:
        updated = {**session, 'title': title}
        write(workbench.root / 'sessions' / sid / 'session.json', updated)
        workbench.sessions[sid] = updated
        return {'session': workbench.public_session(copy.deepcopy(updated))}


def delete_session(workbench, sid):
    from .workbench import write

    with _locked_session(workbench, sid, allow_deleted=True) as session:
        if session is not None:
            write(workbench.root / 'sessions' / sid / 'session.json',
                  {**session, 'deleted_at': time.time()})
            del workbench.sessions[sid]
        # Keep shared locks, databases, memory, raw history and run artifacts.
        return {'ok': True, 'id': sid, 'deleted': True}
