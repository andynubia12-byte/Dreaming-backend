"""Root-scoped macOS generic passwords; secrets never go through files or argv."""
from __future__ import annotations

from contextlib import contextmanager
import ctypes as C
import hashlib
from pathlib import Path
import re
import sys


def validate_api_key(value):
    if not isinstance(value, str):
        raise ValueError('invalid API key')
    key = value.strip()
    if (not 1 <= len(key) <= 8192 or '\n' in value or '\r' in value
            or any(not 32 <= ord(char) <= 126 for char in key)):
        raise ValueError('API key must contain 1..8192 printable ASCII characters without newlines')
    return key


class CredentialStore:
    """set(ref, key), get(ref) -> str | None; delete is used for failed saves only."""

    def __init__(self, root):
        scope = str(Path(root).resolve()).encode('utf-8')
        self.service = 'continuity.model-pool.' + hashlib.sha256(scope).hexdigest()
        self._native = None

    def _libraries(self):
        if self._native is None:
            if sys.platform != 'darwin':
                raise ValueError('persistent API keys require macOS Keychain')
            try:
                security = C.CDLL('/System/Library/Frameworks/Security.framework/Security')
                core = C.CDLL('/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation')
            except OSError:
                raise ValueError('macOS Keychain is unavailable') from None
            pointer, uint = C.c_void_p, C.c_uint32
            signatures = {
                'SecKeychainFindGenericPassword': [pointer, uint, C.c_char_p, uint, C.c_char_p,
                                                  C.POINTER(uint), C.POINTER(pointer), C.POINTER(pointer)],
                'SecKeychainAddGenericPassword': [pointer, uint, C.c_char_p, uint, C.c_char_p,
                                                 uint, pointer, C.POINTER(pointer)],
                'SecKeychainItemModifyAttributesAndData': [pointer, pointer, uint, pointer],
                'SecKeychainItemFreeContent': [pointer, pointer],
                'SecKeychainItemDelete': [pointer],
            }
            for name, arguments in signatures.items():
                function = getattr(security, name)
                function.argtypes, function.restype = arguments, C.c_int32
            core.CFRelease.argtypes, core.CFRelease.restype = [pointer], None
            self._native = security, core
        return self._native

    @staticmethod
    def _check(status):
        if status:
            raise ValueError(f'macOS Keychain operation failed (status {status})')

    @contextmanager
    def _find(self, ref, *, read=False):
        if not isinstance(ref, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', ref):
            raise ValueError('invalid credential reference')
        service, account = self.service.encode(), ref.encode()
        security, core = self._libraries()
        length, data, item = C.c_uint32(), C.c_void_p(), C.c_void_p()
        try:
            status = security.SecKeychainFindGenericPassword(
                None, len(service), service, len(account), account,
                C.byref(length) if read else None, C.byref(data) if read else None, C.byref(item))
            if status != -25300:  # errSecItemNotFound is the only missing-key result.
                self._check(status)
            yield security, item, data, length
        finally:
            if data.value:
                security.SecKeychainItemFreeContent(None, data)
            if item.value:
                core.CFRelease(item)

    def get(self, ref):
        with self._find(ref, read=True) as (_, item, data, length):
            if not item.value:
                return None
            try:
                return C.string_at(data, length.value).decode('utf-8')
            except UnicodeError:
                raise ValueError('invalid API key in macOS Keychain') from None

    def set(self, ref, key):
        password = validate_api_key(key).encode('utf-8')
        with self._find(ref) as (security, item, _, _):
            if item.value:
                self._check(security.SecKeychainItemModifyAttributesAndData(
                    item, None, len(password), password))
            else:
                service, account = self.service.encode(), ref.encode()
                self._check(security.SecKeychainAddGenericPassword(
                    None, len(service), service, len(account), account, len(password), password, None))

    def delete(self, ref):
        with self._find(ref) as (security, item, _, _):
            if item.value:
                self._check(security.SecKeychainItemDelete(item))
