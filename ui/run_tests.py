from unittest import TestLoader, TextTestRunner
from os.path import join, dirname, abspath, isdir


def discover_and_run_suite(path):
    """
    Discover and run a test suite for a given directory.

    Args:
    path (str): The directory path to discover test suites in.

    Raises:
    ValueError: If the provided path is not a string or not a valid directory.
    """
    loader = TestLoader()
    abs_path = abspath(path)
    if not isinstance(path, str):
        raise ValueError(f"Path must be a string. Invalid path: {path}")
    if not isdir(abs_path):
        raise ValueError(f"Path does not exist or is not a directory: {abs_path}")
    
    try:
        suite = loader.discover(abs_path, pattern="*_tests.py")
        runner = TextTestRunner()
        runner.run(suite)
    except Exception as e:
        raise RuntimeError(f"Failed to discover tests in directory {abs_path}: {e}")


if __name__ == "__main__":
    test_dir_field_validators = join(dirname(__file__), 'tests/field_validators_tests')
    test_dir_util = join(dirname(__file__), 'tests/util_tests')
    
    try:
        discover_and_run_suite(test_dir_field_validators)
        discover_and_run_suite(test_dir_util)
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}")
