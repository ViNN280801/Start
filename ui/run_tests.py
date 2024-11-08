import sys
from unittest import TestLoader, TextTestRunner, TextTestResult
from os.path import join, dirname, abspath, isdir, relpath


class CustomTextTestRunner(TextTestRunner):
    def _makeResult(self):
        return CustomTestResult(self.stream, self.descriptions, self.verbosity)


class CustomTestResult(TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.failed_tests = []
    
    def startTest(self, test):
        test_file = relpath(test.__module__.replace('.', '/') + '.py', start=abspath('.'))
        self.stream.write(f"{test_file}: ")
        super().startTest(test)

    def addSuccess(self, test):
        self.stream.write('\033[92m')
        super().addSuccess(test)
        self.stream.write('\033[0m')

    def addFailure(self, test, err):
        self.failed_tests.append(test.id())
        self.stream.write('\033[91m')
        super().addFailure(test, err)
        self.stream.write('\033[0m')

    def addError(self, test, err):
        self.failed_tests.append(test.id())
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
        result = runner.run(suite)
        global total_tests, successful_tests, failed_tests
        total_tests += result.testsRun
        successful_tests += result.testsRun - len(result.failed_tests)
        failed_tests += len(result.failed_tests)
        if result.failed_tests:
            failed_test_names.extend(result.failed_tests)
        return result.wasSuccessful()
    except Exception as e:
        print(f"Failed to discover tests in directory {abs_path}: {e}")
        return False


if __name__ == "__main__":
    total_tests = 0
    successful_tests = 0
    failed_tests = 0
    failed_test_names = []

    test_dirs = [
        join(dirname(__file__), 'tests/field_validators_tests'),
        join(dirname(__file__), 'tests/logger_tests'),
        join(dirname(__file__), 'tests/util_tests')
    ]

    all_tests_passed = True
    for test_dir in test_dirs:
        if not discover_and_run_suite(test_dir):
            all_tests_passed = False

    print(f"\nTotal tests run: {total_tests}")
    print(f"Successful tests: {successful_tests}")
    print(f"Failed tests: {failed_tests}")

    if failed_tests > 0:
        print("\nFailed test names:")
        for test_name in failed_test_names:
            print(f"- {test_name}")

    sys.exit(0 if all_tests_passed else 1)
