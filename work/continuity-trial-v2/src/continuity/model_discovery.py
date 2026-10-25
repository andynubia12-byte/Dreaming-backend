"""Read-only /models discovery for saved profiles or an unsaved connection."""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

from .credential_store import validate_api_key
from .providers import endpoint


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def discover_models(config, pool, body):
    """Accept {model_id} OR {base_url, auth?, api_key_env?, api_key?}; never save."""
    started = time.monotonic()
    key = None
    result = {'ok': False, 'models': [], 'http_status': None, 'url': None}

    def safe(value):
        text = str(value)
        if key:
            text = text.replace(key, '[REDACTED_CREDENTIAL]')
        return text[:800]

    def fail(code, message):
        result['error'] = {'code': code, 'message': safe(message)}
        result['message'] = safe(message)

    try:
        if not isinstance(body, dict):
            raise ValueError('invalid discovery request')
        if 'model_id' in body:
            if set(body) != {'model_id'}:
                raise ValueError('saved discovery accepts only model_id')
            selected = pool.get(body['model_id'])
        else:
            if 'base_url' not in body or set(body) - {'base_url', 'auth', 'api_key_env', 'api_key'}:
                raise ValueError('discovery requires base_url and optional authentication fields')
            selected = body
            if 'api_key' in body:
                key = validate_api_key(body['api_key'])
        auth = selected.get('auth', 'api_key')
        if auth not in ('api_key', 'codex_subscription'):
            raise ValueError('unsupported authentication mode')
        base = selected.get('base_url')
        if not isinstance(base, str) or any(ord(c) < 32 or ord(c) == 127 for c in base):
            raise ValueError('invalid base_url')
        base = base.rstrip('/')
        for suffix in ('/responses', '/chat/completions'):
            if base.endswith(suffix):
                base = base[:-len(suffix)]
                break
        url = endpoint(base, 'responses')[:-len('/responses')] + '/models'
        if auth == 'codex_subscription':
            # No known supported subscription catalog endpoint: do not read auth.json.
            from .codex_credentials import validate_codex_base_url
            validate_codex_base_url(selected['base_url'])
            fail('unsupported', 'Automatic model discovery is unsupported for Codex subscription; enter the model ID manually.')
            return result
        reference = selected.get('api_key_env', getattr(config, 'api_key_env', None))
        if key is None:
            if not isinstance(reference, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', reference):
                raise ValueError('provide an API key or api_key_env')
            key = getattr(pool, '_environ', os.environ).get(reference)
            if not key:
                fail('auth', 'API key is unavailable; configure the credential before discovery.')
                return result
            key = validate_api_key(key)
        if key in url:
            raise ValueError('base_url must not contain an API key')
        result['url'] = url
        request = urllib.request.Request(url, method='GET', headers={'Accept': 'application/json'})
        request.add_unredirected_header('Authorization', 'Bearer ' + key)
        opener = urllib.request.build_opener(_NoRedirect())
        try:
            response = opener.open(request, timeout=15)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            result['http_status'] = response.code
            raw = response.read(1024 * 1024 + 1)
        status = result['http_status']
        if not 200 <= status < 300:
            message = f'HTTP {status}; automatic discovery failed. You can enter the model ID manually.'
            try:
                upstream = json.loads(raw).get('error')
                detail = upstream.get('message') if isinstance(upstream, dict) else upstream
                if isinstance(detail, str):
                    message += ' ' + safe(detail)
            except (ValueError, AttributeError):
                pass
            fail('http_error', message)
            return result
        if len(raw) > 1024 * 1024:
            fail('invalid_response', 'Model catalog response is too large; enter the model ID manually.')
            return result
        try:
            payload = json.loads(raw)
            entries = payload.get('data', payload.get('models')) if isinstance(payload, dict) else None
            if not isinstance(entries, list):
                raise ValueError
            ids = []
            for item in entries:
                model = item if isinstance(item, str) else None
                if isinstance(item, dict):
                    model = next((item[field] for field in ('id', 'slug', 'name')
                                  if isinstance(item.get(field), str)), None)
                if model and len(model) <= 512 and key not in model and model not in ids:
                    ids.append(model)
            if entries and not ids:
                raise ValueError
        except (ValueError, UnicodeError):
            fail('invalid_response', 'Provider did not return a supported model catalog; enter the model ID manually.')
            return result
        result.update(ok=True, models=[{'id': model} for model in ids], error=None,
                      message=f'Found {len(ids)} models.' if ids else 'Provider returned an empty model catalog; manual entry is available.')
    except ValueError:
        fail('invalid_request', 'Invalid model discovery connection or request.')
    except (urllib.error.URLError, OSError, TimeoutError):
        fail('network', 'Model discovery connection failed; enter the model ID manually.')
    finally:
        result['elapsed_seconds'] = round(time.monotonic() - started, 3)
    return result
