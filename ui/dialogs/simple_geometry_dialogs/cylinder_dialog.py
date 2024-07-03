from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QLineEdit,
                             QDialogButtonBox, QMessageBox)
from field_validators import CustomIntValidator, CustomSignedDoubleValidator
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from styles import *
from tabs.graphical_editor.geometry.geometry_limits import *
from tabs.graphical_editor.geometry.geometry_constants import *
from tabs.graphical_editor.geometry.cylinder import Cylinder


class CylinderDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create Cylinder")

        layout = QVBoxLayout(self)
        formLayout = QFormLayout()

        self.xInput = QLineEdit("0.0")
        self.yInput = QLineEdit("0.0")
        self.zInput = QLineEdit("0.0")
        self.radiusInput = QLineEdit("2.5")
        self.dxInput = QLineEdit("5.0")
        self.dyInput = QLineEdit("5.0")
        self.dzInput = QLineEdit("5.0")
        self.resolutionInput = QLineEdit(str(DEFAULT_CYLINDER_RESOLUTION))
        self.meshResolutionInput = QLineEdit(f"{GEOMETRY_MESH_RESOLUTION_VALUE}")

        self.xInput.setValidator(
            CustomSignedDoubleValidator(
                GEOMETRY_CYLINDER_XMIN, GEOMETRY_CYLINDER_XMAX,
                GEOMETRY_CYLINDER_FIELD_PRECISION))
        self.yInput.setValidator(
            CustomSignedDoubleValidator(
                GEOMETRY_CYLINDER_YMIN, GEOMETRY_CYLINDER_YMAX,
                GEOMETRY_CYLINDER_FIELD_PRECISION))
        self.zInput.setValidator(
            CustomSignedDoubleValidator(
                GEOMETRY_CYLINDER_ZMIN, GEOMETRY_CYLINDER_ZMAX,
                GEOMETRY_CYLINDER_FIELD_PRECISION))
        self.radiusInput.setValidator(
            CustomSignedDoubleValidator(
                GEOMETRY_CYLINDER_RADIUS_MIN,
                GEOMETRY_CYLINDER_RADIUS_MAX,
                GEOMETRY_CYLINDER_FIELD_PRECISION))
        self.dxInput.setValidator(
            CustomSignedDoubleValidator(
                GEOMETRY_CYLINDER_DX_MIN,
                GEOMETRY_CYLINDER_DX_MAX,
                GEOMETRY_CYLINDER_FIELD_PRECISION))
        self.dyInput.setValidator(
            CustomSignedDoubleValidator(
                GEOMETRY_CYLINDER_DY_MIN,
                GEOMETRY_CYLINDER_DY_MAX,
                GEOMETRY_CYLINDER_FIELD_PRECISION))
        self.dzInput.setValidator(
            CustomSignedDoubleValidator(
                GEOMETRY_CYLINDER_DZ_MIN,
                GEOMETRY_CYLINDER_DZ_MAX,
                GEOMETRY_CYLINDER_FIELD_PRECISION))
        self.resolutionInput.setValidator(
            CustomIntValidator(GEOMETRY_CYLINDER_MIN_RESOLUTION,
                               GEOMETRY_CYLINDER_MAX_RESOLUTION))
        self.meshResolutionInput.setValidator(
            CustomIntValidator(GEOMETRY_BOX_MESH_RESOLUTION_MIN,
                               GEOMETRY_BOX_MESH_RESOLUTION_MAX))

        self.xInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.yInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.zInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.radiusInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.dxInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.dyInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.dzInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.resolutionInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.resolutionInput.setToolTip(GEOMETRY_CYLINDER_RESOLUTION_HINT)
        self.meshResolutionInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.meshResolutionInput.setToolTip(GEOMETRY_MESH_RESOLUTION_HINT)
        
        self.dxInput.setToolTip("Direction vector component along the X-axis")
        self.dyInput.setToolTip("Direction vector component along the Y-axis")
        self.dzInput.setToolTip("Direction vector component along the Z-axis")

        formLayout.addRow("Base center X:", self.xInput)
        formLayout.addRow("Base center Y:", self.yInput)
        formLayout.addRow("Base center Z:", self.zInput)
        formLayout.addRow("Radius:", self.radiusInput)
        formLayout.addRow("Direction X:", self.dxInput)
        formLayout.addRow("Direction Y:", self.dyInput)
        formLayout.addRow("Direction Z:", self.dzInput)
        formLayout.addRow("Resolution: ", self.resolutionInput)
        formLayout.addRow("Mesh resolution: ", self.meshResolutionInput)

        layout.addLayout(formLayout)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.validate_and_accept)
        self.buttons.rejected.connect(self.reject)

        layout.addWidget(self.buttons)

    def validate_and_accept(self):
        inputs = [
            self.xInput, self.yInput, self.zInput, self.radiusInput,
            self.dxInput, self.dyInput, self.dzInput, self.resolutionInput,
            self.meshResolutionInput
        ]
        all_valid = True

        for input_field in inputs:
            validator = input_field.validator()
            state, _, _ = validator.validate(input_field.text(), 0)

            if isinstance(validator, QDoubleValidator) or isinstance(validator, QIntValidator):
                if state != QDoubleValidator.Acceptable:
                    input_field.setStyleSheet(INVALID_QLINEEDIT_STYLE)
                    all_valid = False
                else:
                    input_field.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)

        if all_valid:
            self.accept()
        else:
            QMessageBox.warning(self, "Invalid input",
                                "Please correct the highlighted fields.")

    def getValues(self):
        values = (float(self.xInput.text()), float(self.yInput.text()), float(self.zInput.text()), 
                  float(self.radiusInput.text()), 
                  float(self.dxInput.text()), float(self.dyInput.text()), float(self.dzInput.text()), 
                  int(self.meshResolutionInput.text()), int(self.resolutionInput.text()))
        return values
    
    def getCylinder(self):
        x, y, z, radius, dx, dy, dz, mesh_resolution, resolution = self.getValues()
        return Cylinder(x, y, z, radius, dx, dy, dz, mesh_resolution, resolution)
