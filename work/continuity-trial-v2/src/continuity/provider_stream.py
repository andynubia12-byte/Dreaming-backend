"""Read provider SSE line-by-line; aggregate the original tool/thinking wire.

Only body deltas and thinking *counts* leave this module through callbacks.
"""
from __future__ import annotations
import json
from .util import looks_like_tool_serialization, tool_serialization_pending_suffix


class StreamProtocolError(RuntimeError):
    pass


class DraftGuard:
    """Hold only suffixes that could become an internal marker across chunks."""
    def __init__(self, callback):
        self.callback = callback
        self.pending = ''
        self.blocked = False

    def feed(self, text, final=False):
        if self.blocked:
            return
        self.pending += text
        if looks_like_tool_serialization(self.pending, final=final):
            self.blocked = True
            self.pending = ''
            self.callback('draft_reset', reason='tool_serialization')
            return
        keep = tool_serialization_pending_suffix(self.pending) if not final else 0
        out = self.pending[:-keep] if keep else self.pending
        self.pending = self.pending[-keep:] if keep else ''
        if out:
            self.callback('answer_delta', text=out)


def read_completion(response, callback, usage):
    message = {'role': 'assistant', 'content': ''}
    calls = {}
    finish = None
    reasoning_chars = 0
    last_notice = -1
    data = []
    done = False

    def consume(raw):
        nonlocal finish, reasoning_chars, last_notice, done
        if raw == '[DONE]':
            done = True
            return
        try:
            chunk = json.loads(raw)
            if chunk.get('usage') is not None:
                usage.update(chunk['usage'])
            if chunk.get('error'):
                raise StreamProtocolError('provider error event')
            for choice in chunk.get('choices') or []:
                if choice.get('index', 0) != 0:
                    continue
                delta = choice.get('delta') or {}
                if finish is not None and any(delta.get(k) for k in ('content', 'reasoning_content', 'tool_calls')):
                    raise StreamProtocolError('delta after finish')
                reason = delta.get('reasoning_content') or ''
                if reason:
                    message['reasoning_content'] = message.get('reasoning_content', '') + reason
                    reasoning_chars += len(reason)
                    bucket = reasoning_chars // 256
                    if bucket != last_notice:
                        callback('thinking', chars=reasoning_chars)
                        last_notice = bucket
                text = delta.get('content') or ''
                if text:
                    message['content'] += text
                    callback('content', text=text)
                for piece in delta.get('tool_calls') or []:
                    index = piece['index']
                    if not isinstance(index, int) or index < 0:
                        raise StreamProtocolError('invalid tool index')
                    call = calls.setdefault(index, {'id': '', 'type': 'function',
                                                   'function': {'name': '', 'arguments': ''}})
                    if piece.get('id'):
                        call['id'] += piece['id']
                    if piece.get('type'):
                        call['type'] = piece['type']
                    for key in ('name', 'arguments'):
                        call['function'][key] += (piece.get('function') or {}).get(key) or ''
                if choice.get('finish_reason') is not None:
                    if finish is not None:
                        raise StreamProtocolError('duplicate finish')
                    finish = choice['finish_reason']
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise StreamProtocolError('invalid provider SSE') from exc

    try:
        # HTTPResponse.readline waits for ONE line, unlike read(large_size).
        for line in response:
            line = line.decode('utf-8').rstrip('\r\n')
            if not line:
                if data:
                    consume('\n'.join(data))
                    data.clear()
                    if done:
                        break
            elif line.startswith('data:'):
                data.append(line[5:].lstrip(' '))
        if data and not done:
            consume('\n'.join(data))
    except (OSError, UnicodeError) as exc:
        raise StreamProtocolError('provider stream disconnected') from exc
    if finish not in ('stop', 'tool_calls', 'length'):
        raise StreamProtocolError('provider stream missing/invalid finish')
    if calls:
        ordered = [calls[k] for k in sorted(calls)]
        if (finish != 'tool_calls' or any(not c['id'] or not c['function']['name']
                or c['type'] != 'function' for c in ordered)
                or len({c['id'] for c in ordered}) != len(ordered)):
            raise StreamProtocolError('incomplete tool call')
        for call in ordered:
            try:
                args = json.loads(call['function']['arguments'])
                if not isinstance(args, dict):
                    raise ValueError()
            except ValueError as exc:
                raise StreamProtocolError('incomplete tool arguments') from exc
        message['tool_calls'] = ordered
    elif finish == 'tool_calls':
        raise StreamProtocolError('missing tool calls')
    return {'choices': [{'index': 0, 'message': message, 'finish_reason': finish}],
            'usage': dict(usage)}
