import unittest
from os.path import join, dirname


if __name__ == "__main__":
    test_dir = join(dirname(__file__), 'field_validators_tests')
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern="*_tests.py")

    runner = unittest.TextTestRunner()
    runner.run(suite)
