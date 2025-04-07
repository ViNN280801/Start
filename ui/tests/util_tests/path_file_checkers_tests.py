import unittest
from unittest.mock import patch, mock_open
from util.path_file_checkers import is_file_valid, is_path_accessible, check_path


class PathFileCheckersTests(unittest.TestCase):
    @patch("util.path_file_checkers.exists")
    @patch("util.path_file_checkers.isfile")
    def test_is_file_valid_path_does_not_exist(self, mock_isfile, mock_exists):
        mock_exists.return_value = False
        mock_isfile.return_value = False
        self.assertFalse(is_file_valid("fake/path/to/file.txt"))

    def test_is_file_valid_empty_path(self):
        self.assertFalse(is_file_valid(""))

    @patch("builtins.open")
    def test_is_path_accessible_file_cannot_be_opened(self, mock_open):
        mock_open.side_effect = IOError
        self.assertFalse(is_path_accessible("fake/path/to/file.txt"))

    @patch("util.path_file_checkers.exists")
    @patch("util.path_file_checkers.isfile")
    def test_is_file_valid_main_cpp(self, mock_isfile, mock_exists):
        mock_exists.return_value = True
        mock_isfile.return_value = True
        self.assertTrue(is_file_valid("../src/main.cpp"))

    @patch("util.path_file_checkers.exists")
    @patch("util.path_file_checkers.isfile")
    def test_is_file_valid_solution_vector_hpp(self, mock_isfile, mock_exists):
        mock_exists.return_value = True
        mock_isfile.return_value = True
        self.assertTrue(
            is_file_valid("../include/FiniteElementMethod/SolutionVector.hpp")
        )

    @patch("builtins.open", new_callable=mock_open)
    def test_is_path_accessible_main_cpp(self, mock_open):
        self.assertTrue(is_path_accessible("../src/main.cpp"))

    @patch("builtins.open", new_callable=mock_open)
    def test_is_path_accessible_solution_vector_hpp(self, mock_open):
        self.assertTrue(
            is_path_accessible("../include/FiniteElementMethod/SolutionVector.hpp")
        )

    @patch("util.path_file_checkers.exists")
    @patch("util.path_file_checkers.isfile")
    def test_check_path_invalid_file(self, mock_isfile, mock_exists):
        mock_exists.return_value = False
        mock_isfile.return_value = False
        with self.assertRaises(ValueError):
            check_path("fake/path/to/file.txt")

    def test_check_path_empty_path(self):
        with self.assertRaises(ValueError):
            check_path("")

    @patch("builtins.open")
    @patch("util.path_file_checkers.exists")
    @patch("util.path_file_checkers.isfile")
    def test_check_path_unaccessible_file(self, mock_isfile, mock_exists, mock_open):
        mock_exists.return_value = True
        mock_isfile.return_value = True
        mock_open.side_effect = IOError
        with self.assertRaises(IOError):
            check_path("fake/path/to/file.txt")

    @patch("builtins.open", new_callable=mock_open)
    @patch("util.path_file_checkers.exists")
    @patch("util.path_file_checkers.isfile")
    def test_check_path_valid_accessible_file(
        self, mock_isfile, mock_exists, mock_open
    ):
        mock_exists.return_value = True
        mock_isfile.return_value = True

        try:
            check_path("../src/main.cpp")
        except Exception as e:
            self.fail(f"check_path raised an exception unexpectedly: {e}")


if __name__ == "__main__":
    unittest.main()
