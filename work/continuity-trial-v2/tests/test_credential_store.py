"""Native calls are mocked: these checks never access the real Keychain."""
import ctypes as C
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from continuity.credential_store import CredentialStore


class CredentialStoreTests(unittest.TestCase):
    def test_service_is_stable_and_isolated_by_managed_root(self):
        self.assertEqual(CredentialStore('/tmp/pool-a').service,
                         CredentialStore('/tmp/pool-a/.').service)
        self.assertNotEqual(CredentialStore('/tmp/pool-a').service,
                            CredentialStore('/tmp/pool-b').service)

    def test_native_get_releases_password_buffer_and_item(self):
        security, core = Mock(), Mock()
        password = C.create_string_buffer(b'synthetic-secret')

        def find(_, service_len, service, account_len, account, length, data, item):
            self.assertEqual(account, b'WORKBENCH_POOL_KEY_TEST')
            C.cast(length, C.POINTER(C.c_uint32))[0] = len(password.value)
            C.cast(data, C.POINTER(C.c_void_p))[0] = C.addressof(password)
            C.cast(item, C.POINTER(C.c_void_p))[0] = 123
            return 0

        security.SecKeychainFindGenericPassword.side_effect = find
        store = CredentialStore('/tmp/offline-pool')
        with patch.object(store, '_libraries', return_value=(security, core)):
            self.assertEqual(store.get('WORKBENCH_POOL_KEY_TEST'), 'synthetic-secret')
        security.SecKeychainItemFreeContent.assert_called_once()
        core.CFRelease.assert_called_once()

    def test_native_set_adds_then_updates_existing_item(self):
        security, core = Mock(), Mock()
        security.SecKeychainFindGenericPassword.return_value = -25300
        security.SecKeychainAddGenericPassword.return_value = 0
        security.SecKeychainItemModifyAttributesAndData.return_value = 0
        store = CredentialStore('/tmp/offline-pool')
        with patch.object(store, '_libraries', return_value=(security, core)):
            self.assertIsNone(store.get('WORKBENCH_POOL_KEY_TEST'))
            store.set('WORKBENCH_POOL_KEY_TEST', 'synthetic-first')
            security.SecKeychainAddGenericPassword.assert_called_once()

            def found(*args):
                C.cast(args[-1], C.POINTER(C.c_void_p))[0] = 123
                return 0

            security.SecKeychainFindGenericPassword.side_effect = found
            store.set('WORKBENCH_POOL_KEY_TEST', 'synthetic-second')
            security.SecKeychainItemModifyAttributesAndData.assert_called_once()
        self.assertEqual(core.CFRelease.call_count, 1)


if __name__ == '__main__':
    unittest.main()
