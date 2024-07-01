from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit,
    QDialogButtonBox, QMessageBox, QComboBox
)
from PyQt5.QtCore import pyqtSignal
from field_validators import CustomSignedDoubleValidator
from vtk import vtkPlane
from styles import *


class CuttingPlaneDialog(QDialog):
    planeUpdated = pyqtSignal(vtkPlane)
    
    def __init__(self, parent=None):
        super (self, parent)
        self.plane = vtkPlane()
        
        self.setWindowTitle("Cutting Plane Configuration")
        layout = QVBoxLayout(self)
        formLayout = QFormLayout()

        # Axis selection
        self.axisSelector = QComboBox()
        self.axisSelector.addItems(["x", "y", "z"])
        self.axisSelector.setToolTip(
            "The axis along which to create the cutting plane. "
            "Options: 'x', 'y', or 'z'."
        )

        # Level input
        self.levelInput = QLineEdit("0.0")
        self.levelInput.setValidator(CustomSignedDoubleValidator(-1e-9, 1e9, 9))
        self.levelInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.levelInput.setToolTip(
            "The position along the specified axis where the cutting plane will be created. "
            "The value should be a real number."
        )

        # Size input
        self.sizeInput = QLineEdit("0.0")
        self.sizeInput.setValidator(CustomSignedDoubleValidator(-1e-9, 1e9, 9))
        self.sizeInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.sizeInput.setToolTip(
            "The half-size of the cutting plane. The cutting plane will have dimensions "
            "of (2 * size, 2 * size) along the other two axes. The value should be a real number."
        )

        # Add rows for the inputs
        formLayout.addRow("Axis:", self.axisSelector)
        formLayout.addRow("Level:", self.levelInput)
        formLayout.addRow("Size:", self.sizeInput)

        layout.addLayout(formLayout)

        # Dialog buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout.addWidget(self.buttons)

    def getValues(self):
        try:
            axis = self.axisSelector.currentText()
            level = float(self.levelInput.text())
            size = float(self.sizeInput.text())
            return axis, level, size
        except Exception as e:
            QMessageBox.warning(self, "Invalid Input", f"Inputs must be valid numbers: {e}")
            return None
        
    def update_plane(self):
        """
        Updates the vtkPlane actor based on the current input values.
        """
        try:
            axis = self.axisSelector.currentText()
            level = float(self.levelInput.text())
            
            if axis == 'x':
                self.plane.SetNormal(1, 0, 0)
                self.plane.SetOrigin(level, 0, 0)
            elif axis == 'y':
                self.plane.SetNormal(0, 1, 0)
                self.plane.SetOrigin(0, level, 0)
            elif axis == 'z':
                self.plane.SetNormal(0, 0, 1)
                self.plane.SetOrigin(0, 0, level)

            self.planeUpdated.emit(self.plane)
        except ValueError as e:
            print(f"Error updating plane: {e}")
