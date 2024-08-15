import unittest
from unittest.mock import patch, mock_open
from util.path_file_checkers import is_file_valid, is_path_accessable


class PathFileCheckersTests(unittest.TestCase):

    @patch('util.path_file_checkers.exists')
    @patch('util.path_file_checkers.isfile')
    def test_is_file_valid_path_does_not_exist(self, mock_isfile, mock_exists):
        mock_exists.return_value = False
        mock_isfile.return_value = False
        self.assertFalse(is_file_valid("fake/path/to/file.txt"))

    @patch('util.path_file_checkers.exists')
    @patch('util.path_file_checkers.isfile')
    def test_is_file_valid_path_exists_but_not_file(self, mock_isfile, mock_exists):
        mock_exists.return_value = True
        mock_isfile.return_value = False
        self.assertFalse(is_file_valid("fake/path/to/directory"))

    @patch('util.path_file_checkers.exists')
    @patch('util.path_file_checkers.isfile')
    def test_is_file_valid_path_exists_and_is_file(self, mock_isfile, mock_exists):
        mock_exists.return_value = True
        mock_isfile.return_value = True
        self.assertTrue(is_file_valid("fake/path/to/file.txt"))

    def test_is_file_valid_empty_path(self):
        self.assertFalse(is_file_valid(""))

    @patch('builtins.open', new_callable=mock_open)
    def test_is_path_accessable_valid_path(self, mock_open):
        self.assertTrue(is_path_accessable("fake/path/to/file.txt"))

    @patch('builtins.open', new_callable=mock_open)
    def test_is_path_accessable_empty_path(self, mock_open):
        self.assertFalse(is_path_accessable(""))

    @patch('builtins.open')
    def test_is_path_accessable_file_cannot_be_opened(self, mock_open):
        mock_open.side_effect = IOError
        self.assertFalse(is_path_accessable("fake/path/to/file.txt"))

    @patch('util.path_file_checkers.exists')
    @patch('util.path_file_checkers.isfile')
    def test_is_file_valid_main_cpp(self, mock_isfile, mock_exists):
        mock_exists.return_value = True
        mock_isfile.return_value = True
        self.assertTrue(is_file_valid("../src/main.cpp"))

    @patch('util.path_file_checkers.exists')
    @patch('util.path_file_checkers.isfile')
    def test_is_file_valid_solution_vector_hpp(self, mock_isfile, mock_exists):
        mock_exists.return_value = True
        mock_isfile.return_value = True
        self.assertTrue(is_file_valid("../include/FiniteElementMethod/SolutionVector.hpp"))

    @patch('builtins.open', new_callable=mock_open)
    def test_is_path_accessable_main_cpp(self, mock_open):
        self.assertTrue(is_path_accessable("../src/main.cpp"))

    @patch('builtins.open', new_callable=mock_open)
    def test_is_path_accessable_solution_vector_hpp(self, mock_open):
        self.assertTrue(is_path_accessable("../include/FiniteElementMethod/SolutionVector.hpp"))


if __name__ == '__main__':
    unittest.main()
