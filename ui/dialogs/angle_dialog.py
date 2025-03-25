from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QLineEdit,
    QDialogButtonBox,
    QMessageBox,
)
from styles import *
from field_validators import CustomSignedDoubleValidator


class AngleDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Change Actor Angle")
        layout = QVBoxLayout(self)
        formLayout = QFormLayout()

        # Input fields for the angles
        self.xAngleInput = QLineEdit("0.0")
        self.yAngleInput = QLineEdit("0.0")
        self.zAngleInput = QLineEdit("0.0")

        self.xAngleInput.setValidator(CustomSignedDoubleValidator(-1e-9, 1e9, 9))
        self.yAngleInput.setValidator(CustomSignedDoubleValidator(-1e-9, 1e9, 9))
        self.zAngleInput.setValidator(CustomSignedDoubleValidator(-1e-9, 1e9, 9))

        self.xAngleInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.yAngleInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.zAngleInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)

        # Add rows for X, Y, and Z angles
        formLayout.addRow("Angle X (degrees):", self.xAngleInput)
        formLayout.addRow("Angle Y (degrees):", self.yAngleInput)
        formLayout.addRow("Angle Z (degrees):", self.zAngleInput)

        layout.addLayout(formLayout)

        # Dialog buttons
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout.addWidget(self.buttons)

    def getValues(self):
        from math import radians

        try:
            x = radians(float(self.xAngleInput.text()))
            y = radians(float(self.yAngleInput.text()))
            z = radians(float(self.zAngleInput.text()))
            return x, y, z
        except Exception as e:
            QMessageBox.warning(
                self, "Invalid Input", f"Angles must be valid numbers: {e}"
            )
            return None
