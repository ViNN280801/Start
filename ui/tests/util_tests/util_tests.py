import io
import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from math import isclose
from util.util import *


class UtilTests(unittest.TestCase):

    def test_rad_to_degree(self):
        self.assertTrue(isclose(rad_to_degree(0), 0))
        self.assertTrue(isclose(rad_to_degree(3.141592653589793), 180))
        self.assertTrue(isclose(rad_to_degree(1.5707963267948966), 90))

    def test_degree_to_rad(self):
        self.assertTrue(isclose(degree_to_rad(0), 0))
        self.assertTrue(isclose(degree_to_rad(180), 3.141592653589793))
        self.assertTrue(isclose(degree_to_rad(90), 1.5707963267948966))

    def test_is_mesh_dims(self):
        self.assertTrue(is_mesh_dims('1'))
        self.assertTrue(is_mesh_dims('2'))
        self.assertTrue(is_mesh_dims('3'))
        self.assertFalse(is_mesh_dims('0'))
        self.assertFalse(is_mesh_dims('4'))
        self.assertFalse(is_mesh_dims('a'))

    @patch('os.path.exists', return_value=True)
    @patch('os.path.isfile', return_value=True)
    def test_is_path_ok_true(self, mock_exists, mock_isfile):
        self.assertTrue(is_path_ok('/some/path/to/file'))

    @patch('os.path.exists', return_value=False)
    @patch('os.path.isfile', return_value=False)
    def test_is_path_ok_false(self, mock_exists, mock_isfile):
        self.assertFalse(is_path_ok('/some/path/to/file'))

    @patch('os.access', return_value=True)
    @patch('os.path.dirname', return_value='/some/path')
    def test_check_path_access_true(self, mock_access, mock_dirname):
        check_path_access('/some/path/to/file')

    @patch('os.access', return_value=False)
    @patch('os.path.dirname', return_value='/some/path')
    def test_check_path_access_false(self, mock_access, mock_dirname):
        with self.assertRaises(OSError):
            check_path_access('/some/path/to/file')

    def test_calculate_direction(self):
        self.assertTrue((calculate_direction([0, 0, 0], [1, 0, 0]) == [1, 0, 0]).all())
        self.assertTrue((calculate_direction([0, 0, 0], [0, 1, 0]) == [0, 1, 0]).all())
        with self.assertRaises(ValueError):
            calculate_direction([0, 0, 0], [0, 0, 0])

    def test_calculate_thetaPhi(self):
        theta, phi = calculate_thetaPhi([0, 0, 0], [1, 0, 0])
        self.assertTrue(isclose(theta, 1.5707963267948966))
        self.assertTrue(isclose(phi, 0))

    def test_calculate_thetaPhi_with_angles(self):
        theta, phi = calculate_thetaPhi_with_angles(1, 0, 0, 0, 0, 0)
        self.assertTrue(isclose(theta, 0.7853981633))
        self.assertTrue(isclose(phi, 0))

    def test_compute_distance_between_points(self):
        num_tests = 1000
        for _ in range(num_tests):
            p1 = np.random.rand(3) * 1000000
            p2 = np.random.rand(3) * 1000000
            expected_distance = np.linalg.norm(p1 - p2)
            self.assertTrue(isclose(compute_distance_between_points(p1, p2), expected_distance))

    @patch('util.gmsh_helpers.gmsh_finalize')
    @patch('sys.exit')
    @patch('os.getpid', return_value=12345)
    def test_signal_handler(self, mock_getpid, mock_exit, mock_finalize):
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            with patch.dict('signal.__dict__', {'SIGINT': 2, 'SIGTERM': 15}):
                with self.assertRaises(SystemExit) as cm:
                    signal_handler(2, None)
                self.assertEqual(cm.exception.code, 1)
    
    @patch('util.gmsh_helpers.gmsh_finalize')
    @patch('sys.exit')
    @patch('os.getpid', return_value=12345)
    def test_signal_handler(self, mock_getpid, mock_exit, mock_finalize):
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            with patch.dict('signal.__dict__', {'SIGINT': 2, 'SIGTERM': 15}):
                with self.assertRaises(SystemExit) as cm:
                    signal_handler(99, None)
                self.assertEqual(cm.exception.code, 1)
    
    @patch('util.gmsh_helpers.gmsh_finalize')
    @patch('sys.exit')
    @patch('os.getpid', return_value=12345)
    def test_signal_handler(self, mock_getpid, mock_exit, mock_finalize):
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            with patch.dict('signal.__dict__', {'SIGINT': 2, 'SIGTERM': 15}):
                with self.assertRaises(SystemExit) as cm:
                    signal_handler(15, None)
                self.assertEqual(cm.exception.code, 1)


if __name__ == '__main__':
    unittest.main()
