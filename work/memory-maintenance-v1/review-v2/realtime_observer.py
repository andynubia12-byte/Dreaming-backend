"""Opt-in sidecar only. Wrap returned model values, never prompts or reasoning.

Installed AFTER the admitted alias wire; return values/exceptions stay identical.
Dedicated inherited FD carries JSONL; a closed observer never affects maintenance.
"""
from functools import wraps
import hashlib
import json
import os
import time


def _text_meta(text, preview=0):
    text = text or ''
    out = {'chars': len(text), 'sha256': hashlib.sha256(text.encode()).hexdigest()}
    if preview:
        out.update(text=text[:preview], truncated=len(text) > preview)
    return out


def _bounded(data):
    wire = json.dumps(data, ensure_ascii=False)
    if len(wire.encode()) > 128 * 1024:
        return {'truncated': True, 'json_preview': wire[:16000],
                'original': _text_meta(wire),
                'note': 'observer limit; full engine result remains in the existing task artifacts'}
    return data


def install(model_classes=None, store_class=None):
    raw_fd = os.environ.get('CONTINUITY_OBSERVER_FD')
    if not raw_fd:
        return
    fd = int(raw_fd)
    if model_classes is None:
        from memmaint.model import DeepSeekModel
        from memmaint.store import Store
        model_classes = (DeepSeekModel,)
        store_class = Store

    def send(phase, **data):
        try:
            wire = (json.dumps({'phase': phase, 'observed_at': time.time(),
                                'data': _bounded(data)}, ensure_ascii=False) + '\n').encode()
            while wire:
                n = os.write(fd, wire)
                wire = wire[n:]
        except (OSError, ValueError, TypeError):
            pass

    def wrap(kind, original):
        @wraps(original)
        def call(self, system, payload):
            if kind == 'propose':
                inputs = {'now': payload.get('now'), 'new_messages': payload.get('new_messages'),
                          'existing_claims_count': len(payload.get('existing_claims') or []),
                          'repair': 'previous_attempt' in payload,
                          'omitted_fields': sorted(set(payload) - {'now', 'new_messages'})}
            elif kind == 'verify':
                inputs = payload
            else:
                inputs = {'now': payload.get('now'),
                          'memory_material': _text_meta(payload.get('memory_material')),
                          'note': 'material omitted; current memory is available through the memory viewer'}
            send('maintenance_' + kind + '_started', input=inputs)
            try:
                result = original(self, system, payload)
            except BaseException as exc:
                send('maintenance_' + kind + '_failed', error_type=type(exc).__name__)
                raise
            output = _text_meta(result, 2000) if kind == 'summarize' else result
            send('maintenance_' + kind + '_result', result=output)
            return result
        return call

    for cls in model_classes:
        for kind in ('propose', 'verify', 'summarize'):
            setattr(cls, kind, wrap(kind, getattr(cls, kind)))
    if store_class:
        original = store_class.publish
        @wraps(original)
        def publish(self, *args, **kwargs):
            send('maintenance_publish_started')
            try:
                result = original(self, *args, **kwargs)
            except BaseException as exc:
                send('maintenance_publish_failed', error_type=type(exc).__name__)
                raise
            send('maintenance_publish_result', manifest={k: v for k, v in result.items()
                 if k in ('version', 'published_at', 'summary_sha', 'memory_sha')})
            return result
        store_class.publish = publish
    send('maintenance_observer_ready', protocol=1)
