from unittest import TestLoader, TextTestRunner, TextTestResult
from os.path import join, dirname, abspath, isdir, relpath


class CustomTextTestRunner(TextTestRunner):
    def _makeResult(self):
        return CustomTestResult(self.stream, self.descriptions, self.verbosity)


class CustomTestResult(TextTestResult):
    def startTest(self, test):
        test_file = relpath(test.__module__.replace('.', '/') + '.py', start=abspath('.'))
        self.stream.write(f"{test_file}: {test._testMethodName} ... ")
        super().startTest(test)

    def addSuccess(self, test):
        self.stream.write('\033[92m')
        super().addSuccess(test)
        self.stream.write('\033[0m')

    def addFailure(self, test, err):
        self.stream.write('\033[91m')
        super().addFailure(test, err)
        self.stream.write('\033[0m')

    def addError(self, test, err):
        self.stream.write('\033[91m')
        super().addError(test, err)
        self.stream.write('\033[0m')

    def addSkip(self, test, reason):
        self.stream.write('\033[93m')
        super().addSkip(test, reason)
        self.stream.write('\033[0m')

    def addExpectedFailure(self, test, err):
        self.stream.write('\033[93m')
        super().addExpectedFailure(test, err)
        self.stream.write('\033[0m')

    def addUnexpectedSuccess(self, test):
        self.stream.write('\033[92m')
        super().addUnexpectedSuccess(test)
        self.stream.write('\033[0m')


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
        runner = CustomTextTestRunner(verbosity=2)
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
