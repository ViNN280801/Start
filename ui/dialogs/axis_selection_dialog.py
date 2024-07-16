from PyQt5.QtWidgets import QDialog, QVBoxLayout, QComboBox, QPushButton


class AxisSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super(AxisSelectionDialog, self).__init__(parent)
        self.setWindowTitle("Select Axis")
        self.setFixedSize(250, 150)
        layout = QVBoxLayout(self)

        self.axisComboBox = QComboBox()
        self.axisComboBox.addItems(["X-axis", "Y-axis", "Z-axis"])
        layout.addWidget(self.axisComboBox)

        okButton = QPushButton("OK")
        okButton.clicked.connect(self.accept)
        layout.addWidget(okButton)

    def getSelectedAxis(self):
        axis_text = self.axisComboBox.currentText()
        if axis_text == "X-axis":
            return 'x'
        elif axis_text == "Y-axis":
            return 'y'
        elif axis_text == "Z-axis":
            return 'z'
