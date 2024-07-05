import unittest
from unittest.mock import patch, mock_open
from util.path_file_chekers import is_file_valid, is_path_accessable


class FileUtilsTests(unittest.TestCase):

    @patch('util.path_file_chekers.exists', return_value=True)
    @patch('util.path_file_chekers.isfile', return_value=True)
    def test_is_file_valid_existing_file(self, mock_exists, mock_isfile):
        self.assertTrue(is_file_valid('dummy_path'))

    @patch('util.path_file_chekers.exists', return_value=False)
    def test_is_file_valid_non_existing_file(self, mock_exists):
        self.assertFalse(is_file_valid('dummy_path'))

    @patch('util.path_file_chekers.exists', return_value=True)
    @patch('util.path_file_chekers.isfile', return_value=False)
    def test_is_file_valid_not_a_file(self, mock_exists, mock_isfile):
        self.assertFalse(is_file_valid('dummy_path'))

    @patch('util.path_file_chekers.exists', return_value=True)
    @patch('util.path_file_chekers.isfile', return_value=True)
    def test_is_file_valid_empty_path(self, mock_exists, mock_isfile):
        self.assertFalse(is_file_valid(''))

    @patch('util.path_file_chekers.exists', return_value=True)
    @patch('util.path_file_chekers.isfile', return_value=True)
    def test_is_file_valid_none_path(self, mock_exists, mock_isfile):
        self.assertFalse(is_file_valid(None))

    @patch('util.path_file_chekers.exists', return_value=True)
    @patch('util.path_file_chekers.isfile', return_value=True)
    def test_is_file_valid_relative_path(self, mock_exists, mock_isfile):
        self.assertTrue(is_file_valid('relative/path/to/file'))

    @patch('util.path_file_chekers.exists', return_value=True)
    @patch('util.path_file_chekers.isfile', return_value=True)
    def test_is_file_valid_absolute_path(self, mock_exists, mock_isfile):
        self.assertTrue(is_file_valid('/absolute/path/to/file'))

    @patch('util.path_file_chekers.exists', return_value=True)
    @patch('util.path_file_chekers.isfile', return_value=True)
    def test_is_file_valid_with_special_characters(self, mock_exists, mock_isfile):
        self.assertTrue(is_file_valid('/path/to/@file#'))

    @patch('util.path_file_chekers.exists', return_value=True)
    @patch('util.path_file_chekers.isfile', return_value=True)
    def test_is_file_valid_long_path(self, mock_exists, mock_isfile):
        long_path = 'a' * 256
        self.assertTrue(is_file_valid(long_path))

    @patch('util.path_file_chekers.exists', return_value=False)
    @patch('util.path_file_chekers.isfile', return_value=False)
    def test_is_file_valid_non_existing_long_path(self, mock_exists, mock_isfile):
        long_path = 'a' * 256
        self.assertFalse(is_file_valid(long_path))

    @patch('builtins.open', new_callable=mock_open)
    def test_is_path_accessable_existing_file(self, mock_file):
        self.assertTrue(is_path_accessable('dummy_path'))

    @patch('builtins.open', side_effect=IOError)
    def test_is_path_accessable_non_existing_file(self, mock_file):
        self.assertFalse(is_path_accessable('dummy_path'))

    @patch('builtins.open', side_effect=IOError)
    def test_is_path_accessable_permission_denied(self, mock_file):
        self.assertFalse(is_path_accessable('/root/protected_file'))

    @patch('builtins.open', new_callable=mock_open)
    def test_is_path_accessable_empty_path(self, mock_file):
        self.assertFalse(is_path_accessable(''))

    @patch('builtins.open', new_callable=mock_open)
    def test_is_path_accessable_none_path(self, mock_file):
        self.assertFalse(is_path_accessable(None))

    @patch('builtins.open', new_callable=mock_open)
    def test_is_path_accessable_relative_path(self, mock_file):
        self.assertTrue(is_path_accessable('relative/path/to/file'))

    @patch('builtins.open', new_callable=mock_open)
    def test_is_path_accessable_absolute_path(self, mock_file):
        self.assertTrue(is_path_accessable('/absolute/path/to/file'))

    @patch('builtins.open', new_callable=mock_open)
    def test_is_path_accessable_with_special_characters(self, mock_file):
        self.assertTrue(is_path_accessable('/path/to/@file#'))

    @patch('builtins.open', new_callable=mock_open)
    def test_is_path_accessable_long_path(self, mock_file):
        long_path = 'a' * 256
        self.assertTrue(is_path_accessable(long_path))

    @patch('builtins.open', side_effect=IOError)
    def test_is_path_accessable_non_existing_long_path(self, mock_file):
        long_path = 'a' * 256
        self.assertFalse(is_path_accessable(long_path))
