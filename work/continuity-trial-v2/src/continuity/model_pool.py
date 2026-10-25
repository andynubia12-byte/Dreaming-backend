"""Shared Responses-only profiles, with optional persistent Keychain API keys.

The backend supplies an approved managed root, outside formal/source data.
Only non-secret metadata is stored in models.json. Profiles are detached snapshots;
deleting or editing a pool entry never changes an existing session's profile.
Subscription availability is deliberately unknown until an actual provider call.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import stat
import threading
import uuid

from .config import PROJECT_ROOT
from .credential_store import CredentialStore, validate_api_key
from .evaluation import _profile

_KEY_PREFIX = 'WORKBENCH_POOL_KEY_'
_LOCKS = {}
_LOCKS_GUARD = threading.Lock()


def _validated(profile, label=None):
    try:
        if not isinstance(profile, dict):
            raise ValueError
        data = copy.deepcopy(profile)
        embedded = data.pop('label', None)
        if data.get('api_format') != 'responses':
            raise ValueError('Responses API is required')
        _profile(data)
        reference = data.get('api_key_env')
        if reference is not None and (not isinstance(reference, str)
                or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', reference)):
            raise ValueError
        label = label if label is not None else embedded if embedded is not None else data['id']
        if not isinstance(label, str) or len(label) > 100:
            raise ValueError
        return {**data, 'label': label}
    except (ValueError, TypeError, AttributeError, OverflowError):
        # Never echo a profile, parameter, label or supplied credential in errors.
        raise ValueError('invalid model profile or label') from None


class ModelPool:
    def __init__(self, root: Path, defaults: list[dict], *, environ=None, credential_store=None):
        self.root = Path(root).absolute()
        self._environ = os.environ if environ is None else environ
        self._credentials = CredentialStore(self.root) if credential_store is None else credential_store
        self._credential_refs = set()
        self._defaults = {}
        self._migrations = {}
        if not isinstance(defaults, list):
            raise ValueError('model defaults must be a list')
        for profile in defaults:
            data = self._stored_profile(profile, 'default')
            if data['id'] in self._defaults:
                raise ValueError('duplicate default model ID')
            self._defaults[data['id']] = data
        with _LOCKS_GUARD:
            self._lock = _LOCKS.setdefault(str(self.root), threading.Lock())
        with self._lock:
            self._check_root()
            try:
                self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            except OSError:
                raise ValueError('model pool directory is unavailable') from None
            self._load()

    def _stored_profile(self, value, source):
        data = copy.deepcopy(value)
        if isinstance(data, dict) and data.get('api_format') == 'chat_completions':
            from .providers import migrate_legacy_profile
            label = data.pop('label', None)
            data = migrate_legacy_profile(data)
            if data.get('id') in self._defaults and 'reasoning' not in data.get('parameters', {}):
                reasoning = self._defaults[data['id']]['parameters'].get('reasoning')
                if reasoning is not None:
                    data.setdefault('parameters', {})['reasoning'] = copy.deepcopy(reasoning)
            data = _validated(data, label)
            self._migrations[(source, data['id'])] = {
                'id': data['id'], 'source': source, 'from': 'chat_completions',
                'to': 'responses', 'scope': 'runtime_copy'}
            return data
        return _validated(data)

    def protocol_migrations(self):
        with self._lock:
            self._load()
            return copy.deepcopy(list(self._migrations.values()))

    def _check_root(self):
        if ('..' in self.root.parts or any(p.is_symlink() for p in (self.root, *self.root.parents))
                or PROJECT_ROOT.is_relative_to(self.root)
                or self.root.is_relative_to(PROJECT_ROOT)
                and not self.root.is_relative_to(PROJECT_ROOT / 'evaluations')
                or any(p.lower().startswith('.env') or p.lower() in
                       {'.codex', '.ssh', 'userprofiles', 'auth.json'} for p in self.root.parts)):
            raise ValueError('model pool requires an isolated managed directory without symlinks or traversal')

    @contextmanager
    def _directory(self):
        self._check_root()
        directory = None
        try:
            directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                info = os.stat('models.json', dir_fd=directory, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError('models.json must be a regular file without symlinks')
            yield directory
        except OSError:
            raise ValueError('model pool storage is unavailable') from None
        finally:
            if directory is not None:
                os.close(directory)

    def _load(self):
        self._migrations = {key: value for key, value in self._migrations.items() if key[0] != 'models.json'}
        with self._directory() as directory:
            try:
                descriptor = os.open('models.json', os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
            except FileNotFoundError:
                self._restore_credentials(self._defaults)
                return {}
            try:
                with os.fdopen(descriptor, 'r', encoding='utf-8') as stream:
                    document = json.load(stream)
                if (not isinstance(document, dict) or set(document) - {'version', 'profiles', 'credential_refs'}
                        or not {'version', 'profiles'} <= set(document)
                        or type(document['version']) is not int or document['version'] != 1
                        or not isinstance(document['profiles'], list)):
                    raise ValueError
                records = {}
                for profile in document['profiles']:
                    data = self._stored_profile(profile, 'models.json')
                    if data['id'] in records:
                        raise ValueError
                    records[data['id']] = data
                references = document.get('credential_refs', [])
                if (not isinstance(references, list) or any(not isinstance(ref, str)
                        or not re.fullmatch(_KEY_PREFIX + r'[A-Za-z0-9_]+', ref) for ref in references)):
                    raise ValueError
            except (ValueError, TypeError, UnicodeError):
                raise ValueError('invalid model pool file') from None
        self._credential_refs.update(references)
        self._restore_credentials({**self._defaults, **records})
        return records

    def _restore_credentials(self, records):
        self._credential_refs.update(p['api_key_env'] for p in records.values()
            if p.get('auth') != 'codex_subscription' and p.get('api_key_env', '').startswith(_KEY_PREFIX))
        for reference in self._credential_refs:
            if not self._environ.get(reference):
                key = self._credentials.get(reference)
                if key is not None:
                    self._environ[reference] = key

    def _persist(self, records):
        with self._directory() as directory:
            temporary = '.models-' + uuid.uuid4().hex + '.tmp'
            try:
                descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                     0o600, dir_fd=directory)
                with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
                    # Retain old references so frozen session profiles survive rotation/deletion.
                    references = self._credential_refs | {p['api_key_env'] for p in records.values()
                        if p.get('auth') != 'codex_subscription'
                        and p.get('api_key_env', '').startswith(_KEY_PREFIX)}
                    json.dump({'version': 1, 'profiles': list(records.values()),
                               'credential_refs': sorted(references)}, stream,
                              ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
                    stream.write('\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, 'models.json', src_dir_fd=directory, dst_dir_fd=directory)
                os.fsync(directory)
            except (OSError, ValueError, TypeError, UnicodeError):
                raise ValueError('could not save model pool') from None
            finally:
                try:
                    os.unlink(temporary, dir_fd=directory)
                except FileNotFoundError:
                    pass

    def profiles(self) -> list[dict]:
        with self._lock:
            return copy.deepcopy(list({**self._defaults, **self._load()}.values()))

    def credential_status(self) -> list[dict]:
        result = []
        for profile in self.profiles():
            if profile.get('auth') == 'codex_subscription':
                available, kind = None, 'codex_subscription'
            else:
                reference = profile['api_key_env']
                available = bool(self._environ.get(reference))
                kind = 'keychain' if reference.startswith(_KEY_PREFIX) else 'environment'
            result.append({'id': profile['id'], 'available': available, 'kind': kind})
        return result

    def get(self, model_id) -> dict:
        if isinstance(model_id, str):
            for profile in self.profiles():
                if profile['id'] == model_id:
                    return profile
        raise ValueError('unknown model ID')

    def save(self, profile: dict, *, label: str | None = None, api_key: str | None = None) -> dict:
        if not isinstance(profile, dict):
            raise ValueError('invalid model profile')
        data = copy.deepcopy(profile)
        key = None
        if api_key is not None:
            if data.get('auth') == 'codex_subscription':
                raise ValueError('subscription profiles cannot use an API key')
            key = validate_api_key(api_key)
        with self._lock:
            if key is not None:
                reference = _KEY_PREFIX + uuid.uuid4().hex.upper()
                while reference in self._environ:
                    reference = _KEY_PREFIX + uuid.uuid4().hex.upper()
                data['api_key_env'] = reference
            records = self._load()
            # Preserve an existing display label on edits that omit it.
            previous = {**self._defaults, **records}.get(data.get('id')) if isinstance(data.get('id'), str) else None
            if label is None and 'label' not in data and previous is not None:
                label = previous['label']
            saved = _validated(data, label)
            records[saved['id']] = saved
            if key is not None:
                self._credentials.set(reference, key)
            try:
                self._persist(records)
            except Exception:
                if key is not None:
                    self._credentials.delete(reference)
                raise
            if key is not None:
                # New references are never reused or removed: old session
                # snapshots keep working across restarts as well.
                self._environ[saved['api_key_env']] = key
            return copy.deepcopy(saved)

    def delete(self, model_id: str) -> None:
        if not isinstance(model_id, str):
            raise ValueError('unknown model ID')
        with self._lock:
            if model_id in self._defaults:
                raise ValueError('default models cannot be deleted; save a copy with a new ID')
            records = self._load()
            if model_id not in records:
                raise ValueError('unknown model ID')
            del records[model_id]
            self._persist(records)
