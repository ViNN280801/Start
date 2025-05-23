from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QLineEdit,
    QDialogButtonBox,
    QMessageBox,
)
from field_validators import CustomIntValidator, CustomSignedDoubleValidator
from PyQt5.QtGui import QDoubleValidator
from styles import *
from tabs.graphical_editor.geometry.geometry_limits import *
from tabs.graphical_editor.geometry.geometry_constants import *
from tabs.graphical_editor.geometry.sphere import Sphere


class SphereDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create Sphere")

        # Create layout
        layout = QVBoxLayout(self)

        # Create form layout for input fields
        formLayout = QFormLayout()

        # Input fields for the sphere parameters
        self.xInput = QLineEdit("0.0")
        self.yInput = QLineEdit("0.0")
        self.zInput = QLineEdit("0.0")
        self.radiusInput = QLineEdit("5.0")
        self.phiResolutionInput = QLineEdit(str(DEFAULT_SPHERE_PHI_RESOLUTION))
        self.thetaResolutionInput = QLineEdit(str(DEFAULT_SPHERE_THETA_RESOLUTION))
        self.meshResolutionInput = QLineEdit(f"{GEOMETRY_MESH_RESOLUTION_SPHERE_VALUE}")

        self.xInput.setValidator(
            CustomSignedDoubleValidator(
                GEOMETRY_SPHERE_XMIN,
                GEOMETRY_SPHERE_XMAX,
                GEOMETRY_SPHERE_FIELD_PRECISION,
            )
        )
        self.yInput.setValidator(
            CustomSignedDoubleValidator(
                GEOMETRY_SPHERE_YMIN,
                GEOMETRY_SPHERE_YMAX,
                GEOMETRY_SPHERE_FIELD_PRECISION,
            )
        )
        self.zInput.setValidator(
            CustomSignedDoubleValidator(
                GEOMETRY_SPHERE_ZMIN,
                GEOMETRY_SPHERE_ZMAX,
                GEOMETRY_SPHERE_FIELD_PRECISION,
            )
        )
        self.radiusInput.setValidator(
            CustomSignedDoubleValidator(
                GEOMETRY_SPHERE_RADIUS_MIN,
                GEOMETRY_SPHERE_RADIUS_MAX,
                GEOMETRY_SPHERE_FIELD_PRECISION,
            )
        )
        self.phiResolutionInput.setValidator(
            CustomIntValidator(
                GEOMETRY_SPHERE_MIN_PHI_RESOLUTION, GEOMETRY_SPHERE_MAX_PHI_RESOLUTION
            )
        )
        self.thetaResolutionInput.setValidator(
            CustomIntValidator(
                GEOMETRY_SPHERE_MIN_THETA_RESOLUTION,
                GEOMETRY_SPHERE_MAX_THETA_RESOLUTION,
            )
        )
        self.meshResolutionInput.setValidator(
            CustomIntValidator(
                GEOMETRY_BOX_MESH_RESOLUTION_MIN, GEOMETRY_BOX_MESH_RESOLUTION_MAX
            )
        )

        self.xInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.yInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.zInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.radiusInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.phiResolutionInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.thetaResolutionInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.meshResolutionInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)

        self.phiResolutionInput.setToolTip(GEOMETRY_SPHERE_PHI_RESOLUTION_HINT)
        self.thetaResolutionInput.setToolTip(GEOMETRY_SPHERE_THETA_RESOLUTION_HINT)
        self.meshResolutionInput.setToolTip(GEOMETRY_MESH_RESOLUTION_HINT)

        formLayout.addRow("Center X:", self.xInput)
        formLayout.addRow("Center Y:", self.yInput)
        formLayout.addRow("Center Z:", self.zInput)
        formLayout.addRow("Radius:", self.radiusInput)
        formLayout.addRow("Mesh resolution: ", self.meshResolutionInput)
        formLayout.addRow("Phi Resolution:", self.phiResolutionInput)
        formLayout.addRow("Theta Resolution:", self.thetaResolutionInput)

        layout.addLayout(formLayout)

        # Dialog buttons
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self
        )
        self.buttons.accepted.connect(self.validate_and_accept)
        self.buttons.rejected.connect(self.reject)

        layout.addWidget(self.buttons)

    def validate_and_accept(self):
        inputs = [
            self.xInput,
            self.yInput,
            self.zInput,
            self.radiusInput,
            self.phiResolutionInput,
            self.thetaResolutionInput,
        ]
        all_valid = True

        for input_field in inputs:
            validator = input_field.validator()
            if isinstance(validator, CustomSignedDoubleValidator):
                if (
                    validator.validate(input_field.text(), 0)[0]
                    != QDoubleValidator.Acceptable
                ):
                    input_field.setStyleSheet(INVALID_QLINEEDIT_STYLE)
                    all_valid = False
                else:
                    input_field.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
            elif isinstance(validator, CustomIntValidator):
                if (
                    validator.validate(input_field.text(), 0)[0]
                    != QDoubleValidator.Acceptable
                ):
                    input_field.setStyleSheet(INVALID_QLINEEDIT_STYLE)
                    all_valid = False
                else:
                    input_field.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)

        if all_valid:
            self.accept()
        else:
            QMessageBox.warning(
                self, "Invalid input", "Please correct the highlighted fields."
            )

    def getValues(self):
        values = (
            float(self.xInput.text()),
            float(self.yInput.text()),
            float(self.zInput.text()),
            float(self.radiusInput.text()),
            int(self.meshResolutionInput.text()),
            int(self.phiResolutionInput.text()),
            int(self.thetaResolutionInput.text()),
        )
        return values

    def getSphere(self):
        (
            x,
            y,
            z,
            radius,
            mesh_resolution,
            phi_resolution,
            theta_resolution,
        ) = self.getValues()
        return Sphere(
            x, y, z, radius, mesh_resolution, phi_resolution, theta_resolution
        )
