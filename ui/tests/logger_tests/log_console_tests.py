import io
import os
import sys
import signal
import unittest
from unittest.mock import patch
from logger.log_console import LogConsole
from util.util import get_cur_datetime


class LogConsoleTests(unittest.TestCase):
    @patch("gmsh.finalize")
    @patch("sys.exit")
    def test_signal_handler_SIGINT(self, mock_exit, mock_finalize):
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            LogConsole.setup_signal_handlers()
            signal_handler = signal.getsignal(signal.SIGINT)
            # Ensure mock_exit is set to raise SystemExit
            mock_exit.side_effect = SystemExit(1)
            with self.assertRaises(SystemExit):
                signal_handler(signal.SIGINT, None)
            output = fake_out.getvalue().strip()
            self.assertIn("Caught signal 2 (SIGINT)", output)
            mock_exit.assert_called_once_with(1)

    @patch("gmsh.finalize")
    @patch("sys.exit")
    def test_signal_handler_unknown_signal(self, mock_exit, mock_finalize):
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            LogConsole.setup_signal_handlers()
            signal_number = signal.SIGUSR1
            signal_handler = signal.getsignal(signal_number)
            mock_exit.side_effect = SystemExit(1)
            with self.assertRaises(SystemExit):
                signal_handler(signal_number, None)
            output = fake_out.getvalue().strip()
            self.assertIn(f"Caught signal {signal_number} (SIGUSR1)", output)
            mock_exit.assert_called_once_with(1)

    @patch("gmsh.finalize")
    @patch("sys.exit")
    def test_crash_supervisor(self, mock_exit, mock_finalize):
        with patch("sys.stderr", new=io.StringIO()) as fake_err:
            with patch("sys.stdout", new=io.StringIO()) as fake_out:
                try:
                    raise ValueError("Test error")
                except ValueError:
                    LogConsole.crash_supervisor(*sys.exc_info())
                output = fake_err.getvalue().strip()

                # The output will include more context
                self.assertIn("Uncaught exception written to the file", output)
                self.assertIn("ValueError: Test error", output)

                # Check that the crash log file was created
                crash_log_file = f"crash_log_{get_cur_datetime()}.txt"
                self.assertTrue(os.path.isfile(crash_log_file))
                if os.path.isfile(crash_log_file):
                    os.remove(crash_log_file)

    def tearDown(self):
        # Cleanup crash log files
        for filename in os.listdir():
            if filename.startswith("crash_log_") and filename.endswith(".txt"):
                os.remove(filename)


if __name__ == "__main__":
    unittest.main()
