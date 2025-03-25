from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QLineEdit,
    QDialogButtonBox,
    QMessageBox,
)
from field_validators import CustomIntValidator, CustomSignedDoubleValidator
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from styles import *
from tabs.graphical_editor.geometry.geometry_limits import *
from tabs.graphical_editor.geometry.geometry_constants import (
    GEOMETRY_MESH_RESOLUTION_HINT,
    GEOMETRY_MESH_RESOLUTION_VALUE,
)
from tabs.graphical_editor.geometry.box import Box


class BoxDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create Box")

        layout = QVBoxLayout(self)
        formLayout = QFormLayout()

        self.xInput = QLineEdit("0.0")
        self.yInput = QLineEdit("0.0")
        self.zInput = QLineEdit("0.0")
        self.lengthInput = QLineEdit("5.0")
        self.widthInput = QLineEdit("5.0")
        self.heightInput = QLineEdit("5.0")
        self.meshResolutionInput = QLineEdit(f"{GEOMETRY_MESH_RESOLUTION_VALUE}")

        self.xInput.setValidator(
            CustomSignedDoubleValidator(
                GEOMETRY_BOX_XMIN, GEOMETRY_BOX_XMAX, GEOMETRY_BOX_FIELD_PRECISION
            )
        )
        self.yInput.setValidator(
            CustomSignedDoubleValidator(
                GEOMETRY_BOX_YMIN, GEOMETRY_BOX_YMAX, GEOMETRY_BOX_FIELD_PRECISION
            )
        )
        self.zInput.setValidator(
            CustomSignedDoubleValidator(
                GEOMETRY_BOX_ZMIN, GEOMETRY_BOX_ZMAX, GEOMETRY_BOX_FIELD_PRECISION
            )
        )
        self.lengthInput.setValidator(
            CustomSignedDoubleValidator(
                GEOMETRY_BOX_LENGTH_MIN,
                GEOMETRY_BOX_LENGTH_MAX,
                GEOMETRY_BOX_FIELD_PRECISION,
            )
        )
        self.widthInput.setValidator(
            CustomSignedDoubleValidator(
                GEOMETRY_BOX_WIDTH_MIN,
                GEOMETRY_BOX_WIDTH_MAX,
                GEOMETRY_BOX_FIELD_PRECISION,
            )
        )
        self.heightInput.setValidator(
            CustomSignedDoubleValidator(
                GEOMETRY_BOX_HEIGHT_MIN,
                GEOMETRY_BOX_HEIGHT_MAX,
                GEOMETRY_BOX_FIELD_PRECISION,
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
        self.lengthInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.widthInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.heightInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.meshResolutionInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.meshResolutionInput.setToolTip(GEOMETRY_MESH_RESOLUTION_HINT)

        formLayout.addRow("Center X:", self.xInput)
        formLayout.addRow("Center Y:", self.yInput)
        formLayout.addRow("Center Z:", self.zInput)
        formLayout.addRow("Length:", self.lengthInput)
        formLayout.addRow("Width:", self.widthInput)
        formLayout.addRow("Height:", self.heightInput)
        formLayout.addRow("Mesh resolution: ", self.meshResolutionInput)

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
            self.lengthInput,
            self.widthInput,
            self.heightInput,
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
            float(self.lengthInput.text()),
            float(self.widthInput.text()),
            float(self.heightInput.text()),
            int(self.meshResolutionInput.text()),
        )

        return values

    def getBox(self):
        x, y, z, length, width, height, mesh_resolution = self.getValues()
        return Box(x, y, z, length, width, height, mesh_resolution)
