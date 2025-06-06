import sys
import platform
import signal
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QPlainTextEdit,
    QTextEdit,
    QWidget,
    QDockWidget,
    QHBoxLayout,
    QApplication,
    QPushButton,
    QLineEdit,
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QTextCharFormat, QTextCursor, QColor, QTextDocument
from util.path_file_checkers import is_file_valid
from util.util import get_cur_datetime, create_secure_tempfile
from .cli_history import CommandLineHistory
from vtk import vtkLogger
from traceback import print_exception
from os import getpid


class LogConsole(QWidget):
    logSignal = pyqtSignal(str)
    runSimulationSignal = pyqtSignal(str)
    uploadMeshSignal = pyqtSignal(str)
    uploadConfigSignal = pyqtSignal(str)
    saveConfigSignal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.setup_ui()
        self.setup_vtk_logger()
        self.setup_gmsh_logger()

        LogConsole.setup_signal_handlers()
        sys.excepthook = self.crash_supervisor

        # Flag to check initial adding of extra new line
        self.isAddedExtraNewLine = False

    def __del__(self):
        try:
            self.cleanup()
        except Exception as e:
            print(f"Error closing up LogConsole resources: {e}")

    def setup_ui(self):
        self.log_console = QPlainTextEdit()
        self.log_console.setReadOnly(True)  # Make the console read-only

        font = self.log_console.font()
        font.setPointSize(12)
        self.log_console.setFont(font)

        self.command_input = CommandLineHistory()
        self.command_input.setPlaceholderText("Enter command...")
        self.command_input.returnPressed.connect(self.handle_command)

        self.layout.addWidget(self.log_console)
        self.layout.addWidget(self.command_input)

        # Add search bar
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search...")
        self.search_input.textChanged.connect(
            self.search_text_in_log
        )  # Connect to real-time search

        self.search_prev_button = QPushButton("Previous")
        self.search_prev_button.clicked.connect(self.search_prev)

        self.search_next_button = QPushButton("Next")
        self.search_next_button.clicked.connect(self.search_next)

        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.search_prev_button)
        search_layout.addWidget(self.search_next_button)

        self.search_container = QWidget()
        self.search_container.setLayout(search_layout)
        self.search_container.setVisible(False)  # Initially hidden

        self.layout.addWidget(self.search_container)

        container = QWidget()
        container.setLayout(self.layout)

        self.setDefaultTextColor(QColor("dark gray"))
        self.log_dock_widget = QDockWidget("Console", self)
        self.log_dock_widget.setWidget(container)
        self.log_dock_widget.setAllowedAreas(Qt.BottomDockWidgetArea)
        self.log_dock_widget.setVisible(True)

    def gmsh_log_monitor(log_file_path, pipe_write_end):
        from time import sleep

        while True:
            with open(log_file_path, "r") as file:
                lines = file.readlines()
            if lines:
                for line in lines:
                    pipe_write_end.send(line.strip())
                with open(log_file_path, "w") as file:
                    pass
            sleep(1)

    def setup_gmsh_logger(self):
        from gmsh import option
        from os import dup, dup2, fdopen

        self.gmsh_log_file_path = create_secure_tempfile()
        self.gmsh_log_file = open(self.gmsh_log_file_path, "w")
        self.appendLog(f"Created log file for the gmsh: {self.gmsh_log_file_path}")

        # Saving original stdout/stderr
        self.original_stdout_fd = dup(1)
        self.original_stderr_fd = dup(2)

        # Redirect stdout/stderr to the log file
        dup2(self.gmsh_log_file.fileno(), 1)
        dup2(self.gmsh_log_file.fileno(), 2)

        option.setNumber("General.Terminal", 1)
        option.setNumber("General.Verbosity", 10)

        self.gmsh_log_file = fdopen(self.gmsh_log_file.fileno(), "w", buffering=1)

        self.start_monitoring_gmsh_log_file()

    def setup_vtk_logger(self):
        from vtk import vtkObject

        vtkObject.GlobalWarningDisplayOn()
        self.vtk_log_file_path = create_secure_tempfile()
        self.appendLog(f"Created log file for the vtk: {self.vtk_log_file_path}")
        vtkLogger.LogToFile(
            self.vtk_log_file_path, vtkLogger.APPEND, vtkLogger.VERBOSITY_INFO
        )
        self.start_monitoring_vtk_log_file()

    def toggle_search(self):
        if self.search_container.isVisible():
            self.search_container.setVisible(False)
        else:
            self.search_container.setVisible(True)
            self.search_input.setFocus()

    def start_monitoring_gmsh_log_file(self):
        self.gmsh_timer = QTimer(self)
        self.gmsh_timer.timeout.connect(self.read_gmsh_log_file)
        self.gmsh_timer.start(1000)

    def read_gmsh_log_file(self):
        with open(self.gmsh_log_file_path, "r") as file:
            for line in file:
                self.appendLog(line.strip())
        open(self.gmsh_log_file_path, "w").close()

    def start_monitoring_vtk_log_file(self):
        # Use QTimer for periodic checks in a GUI-friendly way
        self.vtk_timer = QTimer(self)
        self.vtk_timer.timeout.connect(self.read_vtk_log_file)
        self.vtk_timer.start(1000)  # Check every second

    def read_vtk_log_file(self):
        # Read the log file line by line and append its contents to the log console with appropriate color
        with open(self.vtk_log_file_path, "r") as file:
            for line in file:
                if "WARN|" in line:
                    self.printWarning(line)
                elif "ERR|" in line:
                    self.printError(line)
                else:
                    self.appendLog(line.strip())
        open(self.vtk_log_file_path, "w").close()

        # Adding '\n' to the end of the first output
        if not self.isAddedExtraNewLine:
            logs = self.getAllLogs()
            if logs.endswith("verbosity: 0"):
                self.appendLog("\n")
            self.isAddedExtraNewLine = True

    def cleanup(self):
        from os import remove, dup2

        self.vtk_timer.stop()
        self.gmsh_timer.stop()

        # Reset stdout and stderr to original
        dup2(self.original_stdout_fd, 1)
        dup2(self.original_stderr_fd, 2)

        self.gmsh_log_file.close()
        remove(self.vtk_log_file_path)
        remove(self.gmsh_log_file_path)

        print("cleanup called. files removed")

    def setDefaultTextColor(self, color):
        textFormat = QTextCharFormat()
        textFormat.setForeground(color)

        cursor = self.log_console.textCursor()
        cursor.select(QTextCursor.Document)
        cursor.mergeCharFormat(textFormat)
        cursor.clearSelection()
        self.log_console.setTextCursor(cursor)

    def insert_colored_text(self, prefix: str, message: str, color: str):
        """
        Inserts colored text followed by default-colored text into a QPlainTextEdit widget.

        Parameters:
        - prefix: str, the prefix text to insert in color.
        - message: str, the message text to insert in default color.
        - color: str, the name of the color to use for the prefix.
        """
        cursor = self.log_console.textCursor()

        # Insert colored prefix
        prefix_format = QTextCharFormat()
        prefix_format.setForeground(QColor(color))
        cursor.mergeCharFormat(prefix_format)
        cursor.insertText(prefix, prefix_format)

        # Insert the message in default color
        message_format = QTextCharFormat()
        cursor.setCharFormat(message_format)
        cursor.insertText(message + "\n")

        # Ensure the cursor is moved to the end and reset the format
        cursor.movePosition(cursor.End)
        self.log_console.setTextCursor(cursor)
        self.log_console.setCurrentCharFormat(QTextCharFormat())

    def addNewLine(self):
        self.appendLog("")

    def appendLog(self, message):
        self.log_console.appendPlainText(str(message))

    def printSuccess(self, message):
        self.insert_colored_text("Successfully: ", message, "green")
        self.addNewLine()

    def printError(self, message):
        self.insert_colored_text("Error: ", message, "red")
        self.addNewLine()

    def printInternalError(self, message):
        self.insert_colored_text("Internal error: ", message, "purple")
        self.addNewLine()

    def printWarning(self, message):
        self.insert_colored_text("Warning: ", message, "yellow")
        self.addNewLine()

    def printInfo(self, message):
        self.appendLog("Info: " + str(message))
        self.addNewLine()

    def getAllLogs(self) -> str:
        return self.log_console.toPlainText()

    def handle_command(self):
        """
        Handles commands entered in the command input field.
        """
        command = self.command_input.text().strip().lower()
        self.appendLog(f">> {command}")

        if command:
            self.command_input.history.append(command)
            self.command_input.history_idx = len(self.command_input.history)

        if command == "clear" or command == "clean" or command == "cls":
            self.log_console.clear()
        elif command == "exit" or command == "quit":
            QApplication.quit()
        elif command.startswith("run ") or command.startswith("start "):
            splitted_command = command.split()

            if len(splitted_command) != 2:
                self.appendLog(f"Usage: {splitted_command[0]} <config_name.json>")
                self.command_input.clear()
                return

            configFile = splitted_command[1]
            if not is_file_valid(configFile):
                self.appendLog(f"Invalid or missing file: {configFile}")
                self.command_input.clear()
                return

            self.runSimulationSignal.emit(configFile)

        elif command.startswith("upload mesh "):
            splitted_command = command.split()

            if len(splitted_command) != 3:
                self.appendLog(
                    f"Usage: {splitted_command[0]} {splitted_command[1]} <meshfile.[msh|stp|vtk]>. Make sure that you specify mesh file correctly. Check name of the files and path"
                )
                self.command_input.clear()
                return

            meshFile = splitted_command[2]
            if not is_file_valid(meshFile):
                self.appendLog(f"Invalid or missing file: {meshFile}")
                self.command_input.clear()
                return

            self.uploadMeshSignal.emit(meshFile)

        elif command.startswith("upload config "):
            splitted_command = command.split()

            if len(splitted_command) != 3:
                self.appendLog(
                    f"Usage: {splitted_command[0]} {splitted_command[1]} <config_name.json>"
                )
                self.command_input.clear()
                return

            configFile = splitted_command[2]
            if not is_file_valid(configFile):
                self.appendLog(f"Invalid or missing file: {configFile}")
                self.command_input.clear()
                return

            self.uploadConfigSignal.emit(configFile)

        elif command.startswith("save config "):
            splitted_command = command.split()

            if len(splitted_command) != 3:
                self.appendLog(
                    f"Usage: {splitted_command[0]} {splitted_command[1]} <config_name.json>"
                )
                self.command_input.clear()
                return

            configFile = splitted_command[2]
            self.saveConfigSignal.emit(configFile)

        elif command.strip() == "":
            return
        else:
            self.appendLog(f"Unknown command: {command}")
        self.command_input.clear()

    def search_text_in_log(self):
        search_text = self.search_input.text()
        self.highlight_search_results(search_text)

    def highlight_search_results(self, search_text):
        # Clear existing extra selections
        extra_selections = []

        if search_text:
            cursor = self.log_console.textCursor()
            cursor.beginEditBlock()

            # Move cursor to the beginning
            cursor.movePosition(QTextCursor.Start)

            # Set up the format for highlighting
            format = QTextCharFormat()
            format.setBackground(QColor("purple"))

            # Find and highlight all occurrences
            while True:
                cursor = self.log_console.document().find(search_text, cursor)
                if cursor.isNull():
                    break
                selection = QTextEdit.ExtraSelection()
                selection.cursor = cursor
                selection.format = format
                extra_selections.append(selection)

            cursor.endEditBlock()

        # Apply the extra selections to the log console
        self.log_console.setExtraSelections(extra_selections)

    def search_next(self):
        cursor = self.log_console.textCursor()
        document = self.log_console.document()
        found = document.find(self.search_input.text(), cursor)
        if not found.isNull():
            self.log_console.setTextCursor(found)
        else:
            self.log_console.moveCursor(QTextCursor.Start)
            found = document.find(
                self.search_input.text(), self.log_console.textCursor()
            )
            if not found.isNull():
                self.log_console.setTextCursor(found)

    def search_prev(self):
        cursor = self.log_console.textCursor()
        document = self.log_console.document()
        found = document.find(
            self.search_input.text(), cursor, QTextDocument.FindBackward
        )
        if not found.isNull():
            self.log_console.setTextCursor(found)
        else:
            self.log_console.moveCursor(QTextCursor.End)
            found = document.find(
                self.search_input.text(),
                self.log_console.textCursor(),
                QTextDocument.FindBackward,
            )
            if not found.isNull():
                self.log_console.setTextCursor(found)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F and event.modifiers() == Qt.ControlModifier:
            self.toggle_search()
        else:
            super().keyPressEvent(event)

    @staticmethod
    def signal_handler(signum, frame):
        import gmsh
        from psutil import Process
        from sys import exit
        import signal

        # Mapping signal numbers to signal names
        signals = {
            v: k
            for k, v in signal.__dict__.items()
            if isinstance(v, int) and k.startswith("SIG") and not k.startswith("SIG_")
        }
        signal_name = signals.get(signum, "UNKNOWN")
        process = Process(getpid())
        process_name = " ".join(process.cmdline())
        msg = (
            f"Caught signal {signum} ({signal_name})\n"
            f"Process ID: {getpid()}\n"
            f"Process Name: {process_name}\n"
            f"Frame: {frame}"
        )
        print(msg)

        with open(f"crash_log_{get_cur_datetime()}.txt", "a") as f:
            f.write(msg)

        # Correctly finalize the gmsh
        if gmsh.isInitialized():
            gmsh.finalize()

        exit(1)

    @staticmethod
    def setup_signal_handlers():
        # Define a list of signals to ignore on Unix-like systems
        signals_to_ignore = []
        if platform.system() != "Windows":
            signals_to_ignore = [signal.SIGKILL, signal.SIGSTOP]

        # Handle only the signals supported by the platform
        supported_signals = [
            signal.SIGINT,  # Interrupt from the keyboard
            signal.SIGTERM,  # Termination signal
        ]

        if platform.system() != "Windows":
            # Add Unix-specific signals if not on Windows
            supported_signals.extend(
                [
                    signal.SIGHUP,  # Hangup detected on controlling terminal or death of controlling process
                    signal.SIGQUIT,  # Quit from keyboard
                    signal.SIGUSR1,  # User-defined signal 1
                    signal.SIGUSR2,  # User-defined signal 2
                    signal.SIGPIPE,  # Broken pipe: write to pipe with no readers
                    signal.SIGALRM,  # Alarm clock
                ]
            )

        for sig in supported_signals:
            if sig not in signals_to_ignore:
                try:
                    signal.signal(sig, LogConsole.signal_handler)
                except (ValueError, OSError, RuntimeError) as e:
                    print(f"Cannot catch signal: {sig.name}, {e}")

    @staticmethod
    def crash_supervisor(exc_type, exc_value, exc_traceback):
        # Do not catch keyboard interrupt to allow program termination with Ctrl+C
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        # Log the exception to a file
        crash_filename = f"crash_log_{get_cur_datetime()}.txt"
        with open(crash_filename, "a") as f:
            f.write(f"Uncaught exception:\n")
            print_exception(exc_type, exc_value, exc_traceback, file=f, chain=True)

        print(
            f"Uncaught exception written to the file '{crash_filename}':",
            file=sys.stderr,
        )
        print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr, chain=True)
