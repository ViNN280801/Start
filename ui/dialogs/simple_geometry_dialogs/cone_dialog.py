from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QLineEdit,
    QDialogButtonBox,
    QMessageBox,
)
from field_validators import (
    CustomIntValidator,
    CustomDoubleValidator,
    CustomSignedDoubleValidator,
)
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from styles import *
from tabs.graphical_editor.geometry.geometry_limits import *
from tabs.graphical_editor.geometry.geometry_constants import (
    GEOMETRY_MESH_RESOLUTION_HINT,
    GEOMETRY_MESH_RESOLUTION_VALUE,
    DEFAULT_CONE_RESOLUTION,
)
from tabs.graphical_editor.geometry.cone import Cone


class ConeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create Cone")

        layout = QVBoxLayout(self)
        formLayout = QFormLayout()

        self.xInput = QLineEdit("0.0")
        self.yInput = QLineEdit("0.0")
        self.zInput = QLineEdit("0.0")
        self.dxInput = QLineEdit("1.0")
        self.dyInput = QLineEdit("1.0")
        self.dzInput = QLineEdit("1.0")
        self.radiusInput = QLineEdit("1.0")
        self.resolutionInput = QLineEdit(f"{DEFAULT_CONE_RESOLUTION}")
        self.meshResolutionInput = QLineEdit(f"{GEOMETRY_MESH_RESOLUTION_VALUE}")

        self.xInput.setValidator(
            CustomSignedDoubleValidator(
                GEOMETRY_CONE_XMIN, GEOMETRY_CONE_XMAX, GEOMETRY_CONE_FIELD_PRECISION
            )
        )
        self.yInput.setValidator(
            CustomSignedDoubleValidator(
                GEOMETRY_CONE_YMIN, GEOMETRY_CONE_YMAX, GEOMETRY_CONE_FIELD_PRECISION
            )
        )
        self.zInput.setValidator(
            CustomSignedDoubleValidator(
                GEOMETRY_CONE_ZMIN, GEOMETRY_CONE_ZMAX, GEOMETRY_CONE_FIELD_PRECISION
            )
        )
        self.dxInput.setValidator(
            CustomSignedDoubleValidator(
                GEOMETRY_CONE_DX_MIN,
                GEOMETRY_CONE_DX_MAX,
                GEOMETRY_CONE_FIELD_PRECISION,
            )
        )
        self.dyInput.setValidator(
            CustomSignedDoubleValidator(
                GEOMETRY_CONE_DY_MIN,
                GEOMETRY_CONE_DY_MAX,
                GEOMETRY_CONE_FIELD_PRECISION,
            )
        )
        self.dzInput.setValidator(
            CustomSignedDoubleValidator(
                GEOMETRY_CONE_DZ_MIN,
                GEOMETRY_CONE_DZ_MAX,
                GEOMETRY_CONE_FIELD_PRECISION,
            )
        )
        self.radiusInput.setValidator(
            CustomSignedDoubleValidator(
                GEOMETRY_CONE_R_MIN, GEOMETRY_CONE_R_MAX, GEOMETRY_CONE_FIELD_PRECISION
            )
        )
        self.resolutionInput.setValidator(
            CustomIntValidator(
                GEOMETRY_CONE_RESOLUTION_MIN, GEOMETRY_CONE_RESOLUTION_MAX
            )
        )
        self.meshResolutionInput.setValidator(
            CustomIntValidator(
                GEOMETRY_CONE_MESH_RESOLUTION_MIN, GEOMETRY_CONE_MESH_RESOLUTION_MAX
            )
        )

        self.xInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.yInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.zInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.dxInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.dyInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.dzInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.radiusInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.resolutionInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.meshResolutionInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)

        self.xInput.setToolTip("Enter the X coordinate of the cone's base center.")
        self.yInput.setToolTip("Enter the Y coordinate of the cone's base center.")
        self.zInput.setToolTip("Enter the Z coordinate of the cone's base center.")
        self.dxInput.setToolTip(
            "Enter the X component of the cone's direction vector (determines orientation and height)."
        )
        self.dyInput.setToolTip(
            "Enter the Y component of the cone's direction vector (determines orientation and height)."
        )
        self.dzInput.setToolTip(
            "Enter the Z component of the cone's direction vector (determines orientation and height)."
        )
        self.radiusInput.setToolTip("Enter the radius of the cone's base.")
        self.resolutionInput.setToolTip(
            "Enter the resolution of the cone (number of segments around the cone)."
        )
        self.meshResolutionInput.setToolTip(GEOMETRY_MESH_RESOLUTION_HINT)

        formLayout.addRow("Center X:", self.xInput)
        formLayout.addRow("Center Y:", self.yInput)
        formLayout.addRow("Center Z:", self.zInput)
        formLayout.addRow("Direction X:", self.dxInput)
        formLayout.addRow("Direction Y:", self.dyInput)
        formLayout.addRow("Direction Z:", self.dzInput)
        formLayout.addRow("Radius:", self.radiusInput)
        formLayout.addRow("Cone resolution:", self.resolutionInput)
        formLayout.addRow("Mesh resolution:", self.meshResolutionInput)

        layout.addLayout(formLayout)

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
            self.dxInput,
            self.dyInput,
            self.dzInput,
            self.radiusInput,
            self.resolutionInput,
            self.meshResolutionInput,
        ]
        all_valid = True

        for input_field in inputs:
            validator = input_field.validator()
            state, _, _ = validator.validate(input_field.text(), 0)

            if isinstance(validator, QDoubleValidator) or isinstance(
                validator, QIntValidator
            ):
                if state != QDoubleValidator.Acceptable:
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
            float(self.dxInput.text()),
            float(self.dyInput.text()),
            float(self.dzInput.text()),
            float(self.radiusInput.text()),
            int(self.resolutionInput.text()),
            int(self.meshResolutionInput.text()),
        )
        return values

    def getCone(self):
        x, y, z, dx, dy, dz, radius, resolution, mesh_resolution = self.getValues()
        return Cone(x, y, z, dx, dy, dz, radius, resolution, mesh_resolution)
