"""Pure input metric dependency extracted for the packaged alias wire.

No historical personal driver is imported. Function body is unchanged.
"""
import json


def input_metrics(system, payload):
    """Original driver's estimate and hash, before the subscription envelope."""
    import hashlib
    text = system + '\n' + json.dumps(payload, ensure_ascii=False)
    cjk = sum('\u4e00' <= c <= '\u9fff' for c in text)
    ascii_count = sum(ord(c) < 128 for c in text)
    other_bytes = sum(len(c.encode()) for c in text
                      if ord(c) >= 128 and not '\u4e00' <= c <= '\u9fff')
    estimate = int((cjk + ascii_count / 3 + other_bytes) * 1.15) + 2048
    return {'content_chars': len(text), 'utf8_bytes': len(text.encode()),
            'rough_input_tokens': estimate, 'estimate_is_exact': False,
            'content_sha256': hashlib.sha256(text.encode()).hexdigest()}
