"""Ephemeral, bounded SSE replay. No prompts, credentials or reasoning are stored.

SQLite connections and session locks belong to workers, never subscribers.
"""
from __future__ import annotations

from collections import deque
from contextvars import ContextVar
from contextlib import contextmanager
import json
import threading
import time
import uuid

_CURRENT = ContextVar('continuity_observer', default=None)


class EventHub:
    def __init__(self, max_events=2048, max_bytes=8 * 1024 * 1024):
        self.condition = threading.Condition()
        self.events = deque()
        self.seq = 0
        self.size = 0
        self.max_events = max_events
        self.max_bytes = max_bytes
        self.epoch = uuid.uuid4().hex

    def publish(self, event):
        with self.condition:
            self.seq += 1
            event = {**event, 'seq': self.seq, 'time': time.time(), 'epoch': self.epoch}
            wire = json.dumps(event, ensure_ascii=False)
            size = len(wire.encode('utf-8'))
            if size > self.max_bytes:
                # Explicit observability gap; never silently pretend this is full data.
                event = {**event, 'data': {'omitted': True, 'original_bytes': size,
                         'reason': 'single event exceeds replay byte limit'}}
                wire = json.dumps(event, ensure_ascii=False)
                size = len(wire.encode('utf-8'))
            self.events.append((event, size))
            self.size += size
            while len(self.events) > self.max_events or self.size > self.max_bytes:
                self.size -= self.events.popleft()[1]
            self.condition.notify_all()
            return event

    def read(self, after, *, session_id, request_id=None, channel=None, timeout=10):
        with self.condition:
            if self.seq <= after:
                self.condition.wait(timeout)
            floor = self.events[0][0]['seq'] if self.events else self.seq + 1
            gap = after < floor - 1
            items = [e for e, _ in self.events if e['seq'] > after
                     and e.get('session_id') == session_id
                     and (not request_id or e.get('request_id') == request_id)
                     and (not channel or e.get('channel') == channel)]
            return items, self.seq, gap


class Observer:
    def __init__(self, hub, *, session_id, request_id=None, run_id=None,
                 parent_id=None, tool_call_id=None, task_id=None, channel='chat'):
        self.hub = hub
        self.ids = dict(session_id=session_id, request_id=request_id,
                        run_id=run_id or uuid.uuid4().hex, parent_id=parent_id,
                        tool_call_id=tool_call_id, task_id=task_id, channel=channel)

    def emit(self, phase, **data):
        return self.hub.publish({**self.ids, 'phase': phase, 'data': data})

    def child(self, **ids):
        values = {**self.ids, 'parent_id': self.ids['run_id'],
                  'run_id': uuid.uuid4().hex, **ids}
        return Observer(self.hub, **values)


@contextmanager
def observe(observer):
    token = _CURRENT.set(observer)
    try:
        yield observer
    finally:
        _CURRENT.reset(token)


def current():
    return _CURRENT.get()


def emit(phase, **data):
    observer = current()
    if observer:
        observer.emit(phase, **data)
