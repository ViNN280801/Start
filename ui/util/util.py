import gmsh, tempfile
from vtk import (
    vtkRenderer, vtkPolyData, vtkPolyDataWriter, vtkAppendPolyData,
    vtkPolyDataReader, vtkPolyDataMapper, vtkActor, vtkPolyDataWriter,
    vtkUnstructuredGrid, vtkGeometryFilter
)
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QFormLayout, QLineEdit, QDialogButtonBox, 
    QVBoxLayout, QMessageBox, QPushButton, QTableWidget,
    QTableWidgetItem, QSizePolicy, QLabel, QHBoxLayout,
    QWidget, QScrollArea, QCheckBox
)
from PyQt5.QtCore import QSize
from .converter import is_positive_real_number, is_real_number
from os.path import exists, isfile
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from json import dump, load
from .styles import DEFAULT_QLINEEDIT_STYLE

figure_types = ['Point', 'Line', 'Surface', 'Sphere', 'Box', 'Cylinder', 'Custom']

def is_file_valid(path: str):
    if not exists(path) or not isfile(path) or not path:
        return False
    return True

def is_path_accessable(path):
    try:
        with open(path) as _:
            pass
        return True
    except IOError as _:
        return False
    
def convert_msh_to_vtk(msh_filename: str):
    gmsh.initialize()
    
    vtk_filename = msh_filename.replace('.msh', '.vtk')
    gmsh.open(msh_filename)
    gmsh.write(vtk_filename)
    
    gmsh.finalize()
    return vtk_filename
    
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
        
        self.xAngleInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.yAngleInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.zAngleInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        
        # Add rows for X, Y, and Z angles
        formLayout.addRow("Angle X (degrees):", self.xAngleInput)
        formLayout.addRow("Angle Y (degrees):", self.yAngleInput)
        formLayout.addRow("Angle Z (degrees):", self.zAngleInput)
        
        layout.addLayout(formLayout)
        
        # Dialog buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        
        layout.addWidget(self.buttons)
    
    def getValues(self):
        # Validate inputs and return angles
        x = self.validate_angle(self.xAngleInput.text())
        y = self.validate_angle(self.yAngleInput.text())
        z = self.validate_angle(self.zAngleInput.text())
        
        if x is not None and y is not None and z is not None:
            return x, y, z
        return None

    @staticmethod
    def validate_angle(value: str):
        if is_real_number(value):
            return float(value)
        else:
            QMessageBox.warning(None, "Invalid Input", f"Angle value must be floating point number")
            return None

class MoveActorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Move Actor")

        layout = QVBoxLayout(self)
        formLayout = QFormLayout()

        self.xOffsetInput = QLineEdit("0.0")
        self.yOffsetInput = QLineEdit("0.0")
        self.zOffsetInput = QLineEdit("0.0")
        
        self.xOffsetInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.yOffsetInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.zOffsetInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)

        formLayout.addRow("X Offset:", self.xOffsetInput)
        formLayout.addRow("Y Offset:", self.yOffsetInput)
        formLayout.addRow("Z Offset:", self.zOffsetInput)

        layout.addLayout(formLayout)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout.addWidget(self.buttons)

    def getValues(self):
        try:
            x_offset = float(self.xOffsetInput.text())
            y_offset = float(self.yOffsetInput.text())
            z_offset = float(self.zOffsetInput.text())
            return x_offset, y_offset, z_offset
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Offsets must be valid numbers.")
            return None

class PointDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Create Point")
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Create form layout for input fields
        formLayout = QFormLayout()
        
        # Input fields for the sphere parameters
        self.xInput = QLineEdit("0.0")
        self.yInput = QLineEdit("0.0")
        self.zInput = QLineEdit("0.0")
        
        self.xInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.yInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.zInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        
        formLayout.addRow("Center X:", self.xInput)
        formLayout.addRow("Center Y:", self.yInput)
        formLayout.addRow("Center Z:", self.zInput)
        
        layout.addLayout(formLayout)
        
        # Dialog buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        
        layout.addWidget(self.buttons)
    
    def getValues(self):        
        if not is_real_number(self.xInput.text()):
            QMessageBox.warning(self, "Invalid input", f"{self.xInput.text()} isn't a real number")
            return None
        if not is_real_number(self.yInput.text()):
            QMessageBox.warning(self, "Invalid input", f"{self.yInput.text()} isn't a real number")
            return None
        if not is_real_number(self.zInput.text()):
            QMessageBox.warning(self, "Invalid input", f"{self.zInput.text()} isn't a real number")
            return None
        return float(self.xInput.text()), float(self.yInput.text()), float(self.zInput.text())


class LineDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setWindowTitle("Create Line")

        self.mainLayout = QVBoxLayout(self)  # Main layout for the dialog
        self.scrollArea = QScrollArea(self)  # Scroll area to contain the form
        self.scrollArea.setWidgetResizable(True)  # Allow the contained widget to resize

        # Container widget for the form
        self.containerWidget = QWidget()
        self.formLayout = QFormLayout()  # Form layout for point inputs

        self.inputs = []  # Store all QLineEdit inputs

        # Initialize with 2 points as default for a simple line
        self.add_point_fields()
        self.add_point_fields()

        self.addButton = QPushButton("[+]")
        self.addButton.setFixedSize(QSize(32, 32))
        self.addButton.clicked.connect(self.add_point_fields)

        # Set the form layout to the container widget and add it to the scroll area
        self.containerWidget.setLayout(self.formLayout)
        self.scrollArea.setWidget(self.containerWidget)

        # Add the scroll area and the add button to the main layout
        self.mainLayout.addWidget(self.scrollArea)
        self.mainLayout.addWidget(self.addButton)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        # Add dialog buttons to the main layout
        self.mainLayout.addWidget(self.buttons)

    def add_point_fields(self):
        point_number = len(self.inputs) // 3 + 1
        # Create a horizontal layout for the x, y, z inputs
        hLayout = QHBoxLayout()
        
        x_input = QLineEdit("0.0")
        y_input = QLineEdit("0.0")
        z_input = QLineEdit("0.0")
        
        x_input.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        y_input.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        z_input.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        
        self.inputs.extend([x_input, y_input, z_input])
        
        # Add the inputs to the horizontal layout
        hLayout.addWidget(QLabel(f"Point {point_number} X:"))
        hLayout.addWidget(x_input)
        hLayout.addWidget(QLabel("Y:"))
        hLayout.addWidget(y_input)
        hLayout.addWidget(QLabel("Z:"))
        hLayout.addWidget(z_input)
        
        # Create a container widget for the horizontal layout and add it to the form
        containerWidget = QWidget()
        containerWidget.setLayout(hLayout)
        self.formLayout.addRow(containerWidget)

    def getValues(self):
        if not all(is_real_number(input_field.text()) for input_field in self.inputs):
            QMessageBox.warning(self, "Invalid input", "All coordinates must be real numbers.")
            return None
        return [float(field.text()) for field in self.inputs]

class SurfaceDialog(QDialog):
    def __init__(self, parent=None):       
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setWindowTitle("Create Arbitrary Surface")

        self.mainLayout = QVBoxLayout(self)  # Main layout for the dialog
        self.scrollArea = QScrollArea(self)  # Scroll area to contain the form
        self.scrollArea.setWidgetResizable(True)  # Allow the contained widget to resize

        # Container widget for the form
        self.containerWidget = QWidget()
        self.formLayout = QFormLayout()  # Form layout for point inputs

        self.inputs = []  # Store all QLineEdit inputs

        # Initialize with 3 points
        for _ in range(3):
            self.add_point_fields()

        self.addButton = QPushButton("[+]")
        self.addButton.setFixedSize(QSize(32, 32))
        self.addButton.clicked.connect(self.add_point_fields)

        # Set the form layout to the container widget and add it to the scroll area
        self.containerWidget.setLayout(self.formLayout)
        self.scrollArea.setWidget(self.containerWidget)

        # Add the scroll area and the add button to the main layout
        self.mainLayout.addWidget(self.scrollArea)
        self.mainLayout.addWidget(self.addButton)
        
        self.meshCheckBox = QCheckBox("Mesh object")
        self.meshCheckBox.setChecked(True)
        self.meshCheckBox.stateChanged.connect(self.toggleMeshSizeInput)
        self.meshSizeInput = QLineEdit("1.0")
        self.meshSizeInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.meshSizeLabel = QLabel("Mesh size:")
        meshLayout = QHBoxLayout()
        meshLayout.addWidget(self.meshCheckBox)
        meshLayout.addWidget(self.meshSizeLabel)
        meshLayout.addWidget(self.meshSizeInput)
        self.mainLayout.addLayout(meshLayout)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        # Add dialog buttons to the main layout
        self.mainLayout.addWidget(self.buttons)
    
    def toggleMeshSizeInput(self, state):
        isVisible = state == Qt.Checked
        self.meshSizeInput.setHidden(not isVisible)
        self.meshSizeLabel.setHidden(not isVisible)
        self.meshSizeInput.setEnabled(state == Qt.Checked)

    def add_point_fields(self):
        point_number = len(self.inputs) // 3 + 1
        # Create a horizontal layout for the x, y, z inputs
        hLayout = QHBoxLayout()
        
        x_input = QLineEdit("0.0")
        y_input = QLineEdit("0.0")
        z_input = QLineEdit("0.0")
        
        x_input.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        y_input.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        z_input.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        
        self.inputs.extend([x_input, y_input, z_input])
        
        # Add the inputs to the horizontal layout
        hLayout.addWidget(QLabel(f"Point {point_number} X:"))
        hLayout.addWidget(x_input)
        hLayout.addWidget(QLabel("Y:"))
        hLayout.addWidget(y_input)
        hLayout.addWidget(QLabel("Z:"))
        hLayout.addWidget(z_input)
        
        # Create a container widget for the horizontal layout and add it to the form
        containerWidget = QWidget()
        containerWidget.setLayout(hLayout)
        self.formLayout.addRow(containerWidget)


    def getValues(self):
        if not all(is_real_number(input_field.text()) for input_field in self.inputs):
            QMessageBox.warning(self, "Invalid input", "All coordinates must be real numbers.")
            return None
        values = [float(field.text()) for field in self.inputs]
        
        mesh_size = 1.0
        if self.meshCheckBox.isChecked():
            if not is_real_number(self.meshSizeInput.text()):
                QMessageBox.warning(self, "Invalid input", "Mesh size must be floating point number.")
                return None
            mesh_size = float(self.meshSizeInput.text())
            
        return values, mesh_size, self.meshCheckBox.isChecked()


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
        
        self.xInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.yInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.zInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.radiusInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        
        formLayout.addRow("Center X:", self.xInput)
        formLayout.addRow("Center Y:", self.yInput)
        formLayout.addRow("Center Z:", self.zInput)
        formLayout.addRow("Radius:", self.radiusInput)
        
        layout.addLayout(formLayout)
        
        self.meshCheckBox = QCheckBox("Mesh object")
        self.meshCheckBox.setChecked(True)
        self.meshCheckBox.stateChanged.connect(self.toggleMeshSizeInput)
        self.meshSizeInput = QLineEdit("1.0")
        self.meshSizeInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.meshSizeLabel = QLabel("Mesh size:")
        meshLayout = QHBoxLayout()
        meshLayout.addWidget(self.meshCheckBox)
        meshLayout.addWidget(self.meshSizeLabel)
        meshLayout.addWidget(self.meshSizeInput)
        layout.addLayout(meshLayout)
        
        # Dialog buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        
        layout.addWidget(self.buttons)
        
    def toggleMeshSizeInput(self, state):
        isVisible = state == Qt.Checked
        self.meshSizeInput.setEnabled(state == Qt.Checked)
        self.meshSizeLabel.setHidden(not isVisible)
        self.meshSizeInput.setHidden(not isVisible)
        
    def getValues(self):        
        if not is_real_number(self.xInput.text()):
            QMessageBox.warning(self, "Invalid input", f"{self.xInput.text()} isn't a real number")
            return None
        if not is_real_number(self.yInput.text()):
            QMessageBox.warning(self, "Invalid input", f"{self.yInput.text()} isn't a real number")
            return None
        if not is_real_number(self.zInput.text()):
            QMessageBox.warning(self, "Invalid input", f"{self.zInput.text()} isn't a real number")
            return None
        if not is_positive_real_number(self.radiusInput.text()):
            QMessageBox.warning(self, "Invalid input", f"{self.radiusInput.text()} isn't a real positive number")
            return None
        values = float(self.xInput.text()), float(self.yInput.text()), float(self.zInput.text()), float(self.radiusInput.text())
        
        mesh_size = 0.0
        if self.meshCheckBox.isChecked():
            if not is_real_number(self.meshSizeInput.text()):
                QMessageBox.warning(self, "Invalid input", "Mesh size must be floating point number.")
                return None
            mesh_size = float(self.meshSizeInput.text())
        
        return values, mesh_size, self.meshCheckBox.isChecked()


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
        
        self.xInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.yInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.zInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.lengthInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.widthInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.heightInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        
        formLayout.addRow("Center X:", self.xInput)
        formLayout.addRow("Center Y:", self.yInput)
        formLayout.addRow("Center Z:", self.zInput)
        formLayout.addRow("Length:", self.lengthInput)
        formLayout.addRow("Width:", self.widthInput)
        formLayout.addRow("Height:", self.heightInput)
        
        layout.addLayout(formLayout)
        
        self.meshCheckBox = QCheckBox("Mesh object")
        self.meshCheckBox.setChecked(True)
        self.meshCheckBox.stateChanged.connect(self.toggleMeshSizeInput)
        self.meshSizeInput = QLineEdit("1.0")
        self.meshSizeInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.meshSizeLabel = QLabel("Mesh size:")
        meshLayout = QHBoxLayout()
        meshLayout.addWidget(self.meshCheckBox)
        meshLayout.addWidget(self.meshSizeLabel)
        meshLayout.addWidget(self.meshSizeInput)
        layout.addLayout(meshLayout)
        
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        
        layout.addWidget(self.buttons)
        
    def toggleMeshSizeInput(self, state):
        isVisible = state == Qt.Checked
        self.meshSizeInput.setEnabled(state == Qt.Checked)
        self.meshSizeLabel.setHidden(not isVisible)
        self.meshSizeInput.setHidden(not isVisible)
    
    def getValues(self):
        if not is_real_number(self.xInput.text()):
            QMessageBox.warning(self, "Invalid input", f"{self.xInput.text()} isn't a real number")
            return None
        if not is_real_number(self.yInput.text()):
            QMessageBox.warning(self, "Invalid input", f"{self.yInput.text()} isn't a real number")
            return None
        if not is_real_number(self.zInput.text()):
            QMessageBox.warning(self, "Invalid input", f"{self.zInput.text()} isn't a real number")
            return None
        if not is_positive_real_number(self.lengthInput.text()):
            QMessageBox.warning(self, "Invalid input", f"{self.lengthInput.text()} isn't a real positive number")
            return None
        if not is_positive_real_number(self.widthInput.text()):
            QMessageBox.warning(self, "Invalid input", f"{self.widthInput.text()} isn't a real positive number")
            return None
        if not is_positive_real_number(self.heightInput.text()):
            QMessageBox.warning(self, "Invalid input", f"{self.heightInput.text()} isn't a real positive number")
            return None
        values = (float(self.xInput.text()), float(self.yInput.text()), float(self.zInput.text()),
                float(self.lengthInput.text()), float(self.widthInput.text()), float(self.heightInput.text()))
        
        mesh_size = 1.0
        if self.meshCheckBox.isChecked():
            if not is_real_number(self.meshSizeInput.text()):
                QMessageBox.warning(self, "Invalid input", "Mesh size must be floating point number.")
                return None
            mesh_size = float(self.meshSizeInput.text())

        return values, mesh_size, self.meshCheckBox.isChecked()


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
        
        self.xInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.yInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.zInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.radiusInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.dxInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.dyInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.dzInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        
        formLayout.addRow("Center X:", self.xInput)
        formLayout.addRow("Center Y:", self.yInput)
        formLayout.addRow("Center Z:", self.zInput)
        formLayout.addRow("Radius:", self.radiusInput)
        formLayout.addRow("Length (X-direction):", self.dxInput)
        formLayout.addRow("Width (Y-direction):", self.dyInput)
        formLayout.addRow("Height (Z-direction):", self.dzInput)        
        
        layout.addLayout(formLayout)
        
        self.meshCheckBox = QCheckBox("Mesh object")
        self.meshCheckBox.setChecked(True)
        self.meshCheckBox.stateChanged.connect(self.toggleMeshParamsInput)
        self.meshSizeInput = QLineEdit("1.0")
        self.meshSizeInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        meshLayout = QHBoxLayout()
        meshLayout.addWidget(self.meshCheckBox)
        self.meshSizeLabel = QLabel("Mesh size:")
        meshLayout.addWidget(self.meshSizeLabel)
        meshLayout.addWidget(self.meshSizeInput)
        layout.addLayout(meshLayout)
        
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        
        layout.addWidget(self.buttons)
        
    def toggleMeshParamsInput(self, state):
        isVisible = state == Qt.Checked
        self.meshSizeInput.setEnabled(state == Qt.Checked)
        self.meshSizeLabel.setHidden(not isVisible)
        self.meshSizeInput.setHidden(not isVisible)
    
    def getValues(self):
        if not is_real_number(self.xInput.text()):
            QMessageBox.warning(self, "Invalid input", f"{self.xInput.text()} isn't a real number")
            return None
        if not is_real_number(self.yInput.text()):
            QMessageBox.warning(self, "Invalid input", f"{self.yInput.text()} isn't a real number")
            return None
        if not is_real_number(self.zInput.text()):
            QMessageBox.warning(self, "Invalid input", f"{self.zInput.text()} isn't a real number")
            return None
        if not is_positive_real_number(self.radiusInput.text()):
            QMessageBox.warning(self, "Invalid input", f"{self.radiusInput.text()} isn't a real positive number")
            return None
        if not is_positive_real_number(self.dxInput.text()):
            QMessageBox.warning(self, "Invalid input", f"{self.dxInput.text()} isn't a real positive number")
            return None
        if not is_positive_real_number(self.dyInput.text()):
            QMessageBox.warning(self, "Invalid input", f"{self.dyInput.text()} isn't a real positive number")
            return None
        if not is_positive_real_number(self.dzInput.text()):
            QMessageBox.warning(self, "Invalid input", f"{self.dzInput.text()} isn't a real positive number")
            return None
        values = (float(self.xInput.text()), float(self.yInput.text()), float(self.zInput.text()),
                float(self.radiusInput.text()), float(self.dxInput.text()), float(self.dyInput.text()), float(self.dzInput.text()))
        
        mesh_size = 1.0
        if self.meshCheckBox.isChecked():
            if not is_real_number(self.meshSizeInput.text()):
                QMessageBox.warning(self, "Invalid input", "Mesh size must be floating point number.")
                return None
            mesh_size = float(self.meshSizeInput.text())
            
        return values, mesh_size, self.meshCheckBox.isChecked()

class ShortcutsInfoDialog(QDialog):
    def __init__(self, shortcuts, parent=None):
        super().__init__(parent)
        self.shortcuts = shortcuts
        self.setWindowTitle("Keyboard Shortcuts")
        self.init_ui()


    def init_ui(self):
        self.setMinimumSize(700, 400)
        
        layout = QVBoxLayout(self)
        table = QTableWidget(len(self.shortcuts), 3)
        table.setHorizontalHeaderLabels(["Action", "Shortcut", "Description"])
        table.setEditTriggers(QTableWidget.NoEditTriggers)  # Make the table read-only
        table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        for i, shortcut_info in enumerate(self.shortcuts):
            action, shortcut, description = shortcut_info
            table.setItem(i, 0, QTableWidgetItem(action))
            table.setItem(i, 1, QTableWidgetItem(shortcut))
            table.setItem(i, 2, QTableWidgetItem(description))
        
        table.resizeColumnsToContents()
        layout.addWidget(table)


def align_view_by_axis(axis: str, renderer: vtkRenderer, vtkWidget: QVTKRenderWindowInteractor):
    axis = axis.strip().lower()
        
    if axis not in ['x', 'y', 'z', 'center']:
        return
      
    camera = renderer.GetActiveCamera()
    if axis == 'x':
        camera.SetPosition(1, 0, 0)
        camera.SetViewUp(0, 0, 1)
    elif axis == 'y':
        camera.SetPosition(0, 1, 0)
        camera.SetViewUp(0, 0, 1)
    elif axis == 'z':
        camera.SetPosition(0, 0, 1)
        camera.SetViewUp(0, 1, 0)
    elif axis == 'center':
        camera.SetPosition(1, 1, 1)
        camera.SetViewUp(0, 0, 1)
            
    camera.SetFocalPoint(0, 0, 0)
        
    renderer.ResetCamera()
    vtkWidget.GetRenderWindow().Render()
    

def save_scene(renderer: vtkRenderer, logConsole, fontColor, actors_file='scene_actors.vtk', camera_file='scene_camera.json'):
    if save_actors(renderer, logConsole, fontColor, actors_file) is not None and \
        save_camera_settings(renderer, logConsole, fontColor, camera_file) is not None:
    
        logConsole.insert_colored_text('Successfully: ', 'green')
        logConsole.insert_colored_text(f'Saved scene from to the files: {actors_file} and {camera_file}\n', fontColor)
    

def save_actors(renderer: vtkRenderer, logConsole, fontColor, actors_file='scene_actors.vtk'):
    try:
        append_filter = vtkAppendPolyData()
        actors_collection = renderer.GetActors()
        actors_collection.InitTraversal()
        
        for i in range(actors_collection.GetNumberOfItems()):
            actor = actors_collection.GetNextActor()
            if actor.GetMapper() and actor.GetMapper().GetInput():
                poly_data = actor.GetMapper().GetInput()
                if isinstance(poly_data, vtkPolyData):
                    append_filter.AddInputData(poly_data)
        
        append_filter.Update()

        writer = vtkPolyDataWriter()
        writer.SetFileName(actors_file)
        writer.SetInputData(append_filter.GetOutput())
        writer.Write()

        logConsole.insert_colored_text('Info: ', 'blue')
        logConsole.insert_colored_text(f'Saved all actors to {actors_file}\n', fontColor)
        return 1
    except Exception as e:
        logConsole.insert_colored_text('Error: ', 'red')
        logConsole.insert_colored_text(f'Failed to save actors: {e}\n', fontColor)
        return None
        
        
def save_camera_settings(renderer: vtkRenderer, logConsole, fontColor, camera_file='scene_camera.json'):
    try:
        camera = renderer.GetActiveCamera()
        camera_settings = {
            'position': camera.GetPosition(),
            'focal_point': camera.GetFocalPoint(),
            'view_up': camera.GetViewUp(),
            'clip_range': camera.GetClippingRange(),
        }
        with open(camera_file, 'w') as f:
            dump(camera_settings, f)

        return 1
    except Exception as e:
        logConsole.insert_colored_text('Error: ', 'red')
        logConsole.insert_colored_text(f'Failed to save camera settings: {e}\n', fontColor)
        return None
        

def load_scene(vtkWidget: QVTKRenderWindowInteractor, renderer: vtkRenderer, logConsole, fontColor, actors_file='scene_actors.vtk', camera_file='scene_camera.json'):
    if load_actors(renderer, logConsole, fontColor, actors_file) is not None and \
        load_camera_settings(renderer, logConsole, fontColor, camera_file) is not None:
    
        vtkWidget.GetRenderWindow().Render()
        logConsole.insert_colored_text('Successfully: ', 'green')
        logConsole.insert_colored_text(f'Loaded scene from the files: {actors_file} and {camera_file}\n', fontColor)


def load_actors(renderer: vtkRenderer, logConsole, fontColor, actors_file='scene_actors.vtk'):
    try:
        reader = vtkPolyDataReader()
        reader.SetFileName(actors_file)
        reader.Update()
        
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(reader.GetOutput())
        
        actor = vtkActor()
        actor.SetMapper(mapper)
        renderer.AddActor(actor)
        renderer.ResetCamera()

        logConsole.insert_colored_text('Info: ', 'blue')
        logConsole.insert_colored_text(f'Loaded actors from {actors_file}\n', fontColor)
        return 1
    except Exception as e:
        logConsole.insert_colored_text('Error: ', 'red')
        logConsole.insert_colored_text(f'Failed to load actors: {e}\n', fontColor)
        return None
        
        
def load_camera_settings(renderer: vtkRenderer, logConsole, fontColor, camera_file='scene_camera.json'):
    try:
        with open(camera_file, 'r') as f:
            camera_settings = load(f)
        
        camera = renderer.GetActiveCamera()
        camera.SetPosition(*camera_settings['position'])
        camera.SetFocalPoint(*camera_settings['focal_point'])
        camera.SetViewUp(*camera_settings['view_up'])
        camera.SetClippingRange(*camera_settings['clip_range'])
        
        renderer.ResetCamera()
        return 1
    except Exception as e:
        logConsole.insert_colored_text('Error: ', 'red')
        logConsole.insert_colored_text(f'Failed to load camera settings: {e}\n', fontColor)
        return None

def get_polydata_from_actor(actor: vtkActor):
    mapper = actor.GetMapper()
    if hasattr(mapper, "GetInput"):
        return mapper.GetInput()
    else:
        return None


def write_vtk_polydata_to_file(polyData):
    writer = vtkPolyDataWriter()
    writer.SetInputData(polyData)

    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.vtk')
    temp_file_name = temp_file.name
    temp_file.close()

    # Set the filename in the writer and write
    writer.SetFileName(temp_file_name)
    writer.Write()

    # Return the path to the temporary file
    return temp_file_name

def is_conversion_success(polyData):
    # Check if the polyData is not None
    if polyData is None:
        return False

    # Check if there are any points and cells in the polyData
    numberOfPoints = polyData.GetNumberOfPoints()
    numberOfCells = polyData.GetNumberOfCells()

    if numberOfPoints > 0 and numberOfCells > 0:
        return True # Conversion was successful and resulted in a non-empty polyData
    else:
        return False # Conversion failed to produce meaningful polyData


def convert_vtkUnstructuredGrid_to_vtkPolyData_helper(ugrid: vtkUnstructuredGrid):
    geometryFilter = vtkGeometryFilter()
    geometryFilter.SetInputData(ugrid)
    
    geometryFilter.Update()
    
    polyData = geometryFilter.GetOutput()
    if not is_conversion_success(polyData):
        return None
    
    return polyData

def convert_vtkUnstructuredGrid_to_vtkPolyData(data):
    if data.IsA("vtkUnstructuredGrid"):
        return convert_vtkUnstructuredGrid_to_vtkPolyData_helper(data)
    elif data.IsA("vtkPolyData"):
        return data
    else:
        return None
    
