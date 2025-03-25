from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QLineEdit,
    QDialogButtonBox,
    QMessageBox,
)
from styles import *
from field_validators import CustomDoubleValidator


class ScaleDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Change Actor Scaling")

        layout = QVBoxLayout(self)

        formLayout = QFormLayout()

        # Input fields for the scaling factors
        self.dxInput = QLineEdit("1.0")
        self.dyInput = QLineEdit("1.0")
        self.dzInput = QLineEdit("1.0")

        self.dxInput.setValidator(CustomDoubleValidator(1e-9, 1e9, 9))
        self.dyInput.setValidator(CustomDoubleValidator(1e-9, 1e9, 9))
        self.dzInput.setValidator(CustomDoubleValidator(1e-9, 1e9, 9))

        self.dxInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.dyInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.dzInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)

        # Add rows for X, Y, and Z scaling factors
        formLayout.addRow("Scale X:", self.dxInput)
        formLayout.addRow("Scale Y:", self.dyInput)
        formLayout.addRow("Scale Z:", self.dzInput)

        layout.addLayout(formLayout)

        # Dialog buttons
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout.addWidget(self.buttons)

    def getValues(self):
        try:
            x_offset = float(self.dxInput.text())
            y_offset = float(self.dyInput.text())
            z_offset = float(self.dzInput.text())
            return x_offset, y_offset, z_offset
        except Exception as e:
            QMessageBox.warning(
                self, "Invalid Input", f"Offsets must be valid numbers: {e}"
            )
            return None
