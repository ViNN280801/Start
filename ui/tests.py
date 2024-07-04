from unittest import TestLoader, TextTestRunner
from os.path import join, dirname


if __name__ == "__main__":
    test_dir = join(dirname(__file__), 'tests/field_validators_tests')
    loader = TestLoader()
    suite = loader.discover(test_dir, pattern="*_tests.py")

    runner = TextTestRunner()
    runner.run(suite)
