import io
import os
import sys
import signal
import unittest
from unittest.mock import patch
from logger.log_console import LogConsole
from util.util import get_cur_datetime


class LogConsoleTests(unittest.TestCase):

    @patch('gmsh.finalize')
    @patch('sys.exit')
    def test_signal_handler_SIGINT(self, mock_exit, mock_finalize):
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            LogConsole.setup_signal_handlers()
            signal_handler = signal.getsignal(signal.SIGINT)
            with self.assertRaises(SystemExit) as cm:
                signal_handler(signal.SIGINT, None)
            self.assertEqual(cm.exception.code, 1)
            output = fake_out.getvalue().strip()
            self.assertIn("Caught signal 2 (SIGINT)", output)
    
    @patch('gmsh.finalize')
    @patch('sys.exit')
    def test_signal_handler_unknown_signal(self, mock_exit, mock_finalize):
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            LogConsole.setup_signal_handlers()
            # Using a valid but uncommon signal number
            signal_number = signal.SIGUSR1
            signal_handler = signal.getsignal(signal_number)
            with self.assertRaises(SystemExit) as cm:
                signal_handler(signal_number, None)
            self.assertEqual(cm.exception.code, 1)
            output = fake_out.getvalue().strip()
            self.assertIn(f"Caught signal {signal_number} (SIGUSR1)", output)
    
    @patch('gmsh.finalize')
    @patch('sys.exit')
    def test_signal_handler_all_signals(self, mock_exit, mock_finalize):
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            LogConsole.setup_signal_handlers()
            signal_aliases = {v: k for k, v in signal.__dict__.items() if isinstance(v, int) and k.startswith('SIG') and not k.startswith('SIG_')}
            for sig_name, sig in signal.__dict__.items():
                if isinstance(sig, int) and sig_name.startswith('SIG') and not sig_name.startswith('SIG_'):
                    try:
                        with self.assertRaises(SystemExit) as cm:
                            LogConsole.signal_handler(sig, None)
                        self.assertEqual(cm.exception.code, 1)
                        output = fake_out.getvalue().strip()
                        expected_sig_name = signal_aliases.get(sig, sig_name)
                        self.assertIn(f"Caught signal {sig} ({expected_sig_name})", output)
                        fake_out.truncate(0)
                        fake_out.seek(0)
                    except (TypeError, ValueError, OSError) as e:
                        print(f"Skipping signal {sig_name} ({sig}): {e}")


    @patch('gmsh.finalize')
    @patch('sys.exit')
    def test_crash_supervisor(self, mock_exit, mock_finalize):
        with patch('sys.stderr', new=io.StringIO()) as fake_err:
            with patch('sys.stdout', new=io.StringIO()) as fake_out:
                try:
                    raise ValueError("Test error")
                except ValueError:
                    LogConsole.crash_supervisor(*sys.exc_info())
                output = fake_err.getvalue().strip()
                self.assertIn("Uncaught exception:", output)
                self.assertIn("ValueError: Test error", output)

                crash_log_file = f"crash_log_{get_cur_datetime()}.txt"
                self.assertTrue(os.path.isfile(crash_log_file))
                if os.path.isfile(crash_log_file):
                    os.remove(crash_log_file)

    def tearDown(self):
        # Cleanup crash log files
        for filename in os.listdir():
            if filename.startswith("crash_log_") and filename.endswith(".txt"):
                os.remove(filename)


if __name__ == '__main__':
    unittest.main()
