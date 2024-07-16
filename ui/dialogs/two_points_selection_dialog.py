from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLineEdit, QLabel, QDialogButtonBox
from field_validators import CustomSignedDoubleValidator
from styles import DEFAULT_QLINEEDIT_STYLE


class TwoPointSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Two Points for Plane")
        self.setGeometry(100, 100, 300, 200)
        layout = QVBoxLayout()

        self.point1_x = QLineEdit(self)
        self.point1_y = QLineEdit(self)
        self.point1_z = QLineEdit(self)
        self.point2_x = QLineEdit(self)
        self.point2_y = QLineEdit(self)
        self.point2_z = QLineEdit(self)
        
        self.point1_x.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.point1_y.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.point1_z.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.point2_x.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.point2_y.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.point2_z.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        
        self.point1_x.setValidator(CustomSignedDoubleValidator(-1e9, 1e9, 9))
        self.point1_y.setValidator(CustomSignedDoubleValidator(-1e9, 1e9, 9))
        self.point1_z.setValidator(CustomSignedDoubleValidator(-1e9, 1e9, 9))
        self.point2_x.setValidator(CustomSignedDoubleValidator(-1e9, 1e9, 9))
        self.point2_x.setValidator(CustomSignedDoubleValidator(-1e9, 1e9, 9))
        self.point2_x.setValidator(CustomSignedDoubleValidator(-1e9, 1e9, 9))

        layout.addWidget(QLabel("Point 1:"))
        layout.addWidget(QLabel("X:"))
        layout.addWidget(self.point1_x)
        layout.addWidget(QLabel("Y:"))
        layout.addWidget(self.point1_y)
        layout.addWidget(QLabel("Z:"))
        layout.addWidget(self.point1_z)

        layout.addWidget(QLabel("Point 2:"))
        layout.addWidget(QLabel("X:"))
        layout.addWidget(self.point2_x)
        layout.addWidget(QLabel("Y:"))
        layout.addWidget(self.point2_y)
        layout.addWidget(QLabel("Z:"))
        layout.addWidget(self.point2_z)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def getPoints(self):
        point1 = (float(self.point1_x.text()), float(self.point1_y.text()), float(self.point1_z.text()))
        point2 = (float(self.point2_x.text()), float(self.point2_y.text()), float(self.point2_z.text()))
        return point1, point2
