from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit,
    QDialogButtonBox, QMessageBox, QComboBox
)
from PyQt5.QtCore import pyqtSignal
from vtk import (
    vtkPlaneSource, vtkActor, vtkPolyDataMapper
)
from field_validators import CustomSignedDoubleValidator
from util.vtk_helpers import remove_actor, add_actor
from logger.internal_logger import InternalLogger
from tabs.graphical_editor.geometry.vtk_geometry.vtk_geometry_manipulator import VTKGeometryManipulator
from styles import *


class CuttingPlaneDialog(QDialog):
    accepted_with_values = pyqtSignal(str, float, float)

    def __init__(self, parent=None, vtkWidget=None, renderer=None):
        super().__init__(parent)
        self.geditor = parent
        self.vtkWidget = vtkWidget
        self.renderer = renderer
        self.axis = 'z'
        self.cutting_plane_actor = None

        self.updating = False
        self.rotation_axis = 'X'
        self.move_axis = 'z'

        self.setMinimumWidth(250)
        self.setMaximumWidth(250)

        self.setWindowTitle("Cutting Plane Configuration")
        layout = QVBoxLayout(self)
        formLayout = QFormLayout()

        # Axis selection
        self.planeSelector = QComboBox()
        self.planeSelector.addItems(["XY", "YZ", "XZ"])
        self.planeSelector.setToolTip(
            "Select the plane. "
            "Options: 'XY', 'YZ', or 'XZ'."
        )
        self.planeSelector.currentIndexChanged.connect(self.update_and_refresh)

        # Level input
        self.levelInput = QLineEdit("0.0")
        self.levelInput.setValidator(CustomSignedDoubleValidator(-1e9, 1e9, 9))
        self.levelInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.levelInput.setToolTip(
            "The position along the specified axis where the cutting plane will be created. "
            "Plane - Axis configurations: "
            "  XY  ->  Z  "
            "  XZ  ->  Y  "
            "  YZ  ->  X  "
            "The value should be a real number."
        )
        self.levelInput.textChanged.connect(self.update_level_from_text)

        # Angle input
        self.angleInput = QLineEdit("0.0")
        self.angleInput.setValidator(CustomSignedDoubleValidator(-360, 360, 3))
        self.angleInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.angleInput.setToolTip(
            "The angle of the cutting plane in degrees. The value should be a real number."
        )
        self.angleInput.textChanged.connect(self.update_angle_from_text)
        
        # Rotation axis selection
        self.rotationAxisSelector = QComboBox()
        self.rotationAxisSelector.setToolTip(
            "Select the axis around which to rotate the cutting plane. "
        )
        self.rotationAxisSelector.currentIndexChanged.connect(self.update_rotation_axis)

        # User size input
        self.userSizeInput = QLineEdit("50.0")
        self.userSizeInput.setValidator(CustomSignedDoubleValidator(-1e9, 1e9, 9))
        self.userSizeInput.setStyleSheet(DEFAULT_QLINEEDIT_STYLE)
        self.userSizeInput.setToolTip(
            "The size of the cutting plane for visualization. The value should be a real number."
        )
        self.userSizeInput.textChanged.connect(self.update_user_size_from_text)

        # Add rows for the inputs
        formLayout.addRow("Plane:", self.planeSelector)
        formLayout.addRow("Rotation Axis:", self.rotationAxisSelector)
        formLayout.addRow("Level:", self.levelInput)
        formLayout.addRow("Angle (in degrees):", self.angleInput)
        formLayout.addRow("Size:", self.userSizeInput)

        layout.addLayout(formLayout)

        # Dialog buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.on_accept)
        self.buttons.rejected.connect(self.on_reject)

        layout.addWidget(self.buttons)
        self.update_and_refresh()
    
    def update_rotation_axis_options(self):
        current_plane = self.planeSelector.currentText()
        self.rotationAxisSelector.blockSignals(True)
        self.rotationAxisSelector.clear()
        if current_plane == 'XY':
            self.rotationAxisSelector.addItems(['X', 'Y'])
        elif current_plane == 'YZ':
            self.rotationAxisSelector.addItems(['Y', 'Z'])
        elif current_plane == 'XZ':
            self.rotationAxisSelector.addItems(['X', 'Z'])
        self.rotationAxisSelector.blockSignals(False)
        self.rotation_axis = self.rotationAxisSelector.currentText()

    def update_rotation_axis(self):
        self.rotation_axis = self.rotationAxisSelector.currentText()

    def update_user_size_from_text(self):
        try:
            size = float(self.userSizeInput.text())
            self.update_and_refresh()
        except ValueError:
            pass

    def update_level_from_text(self):
        try:
            level = float(self.levelInput.text())
            self.update_and_refresh()
        except ValueError:
            pass

    def update_angle_from_text(self):
        try:
            angle = float(self.angleInput.text())
            self.update_and_refresh()
        except ValueError:
            pass

    def update_and_refresh(self):
        if self.updating:
            return

        self.updating = True
        self.updateAxis()
        try:
            axis, level, angle = self.getValues()
            user_size = float(self.userSizeInput.text())
            if self.move_axis == 'x':
                origin = [level, -user_size, -user_size]
                point1 = [level, user_size, -user_size]
                point2 = [level, -user_size, user_size]
            elif self.move_axis == 'y':
                origin = [-user_size, level, -user_size]
                point1 = [user_size, level, -user_size]
                point2 = [-user_size, level, user_size]
            elif self.move_axis == 'z':
                origin = [-user_size, -user_size, level]
                point1 = [user_size, -user_size, level]
                point2 = [-user_size, user_size, level]
            self.manage_plane_actor(level, angle, origin, point1, point2)
        except ValueError:
            pass
        self.updating = False

    def updateAxis(self):
        if self.planeSelector.currentText() == 'XY':
            self.move_axis = 'z'
        elif self.planeSelector.currentText() == 'YZ':
            self.move_axis = 'x'
        elif self.planeSelector.currentText() == 'XZ':
            self.move_axis = 'y'
        else:
            raise ValueError(f"{InternalLogger.pretty_function_details()}: There is no such plane configuration: '{self.planeSelector.currentText()}'")

        self.update_rotation_axis_options()

    def manage_plane_actor(self, level: float, angle: float, origin, point1, point2):
        """
        Updates the plane actor based on the vtkPlane, size, and angle, and adds it to the renderer.
        """
        remove_actor(self.vtkWidget, self.renderer, self.cutting_plane_actor, needResetCamera=False)

        planeSource = vtkPlaneSource()
        planeSource.SetOrigin(origin)
        planeSource.SetPoint1(point1)
        planeSource.SetPoint2(point2)
        planeSource.Update()

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(planeSource.GetOutputPort())

        self.cutting_plane_actor = vtkActor()
        self.cutting_plane_actor.SetMapper(mapper)
        self.cutting_plane_actor.GetProperty().SetColor(1, 0, 0)

        if self.rotation_axis == 'X':
            VTKGeometryManipulator.rotate(self.cutting_plane_actor, angle, 0, 0)
        elif self.rotation_axis == 'Y':
            VTKGeometryManipulator.rotate(self.cutting_plane_actor, 0, angle, 0)
        elif self.rotation_axis == 'Z':
            VTKGeometryManipulator.rotate(self.cutting_plane_actor, 0, 0, 0, angle)

        add_actor(self.vtkWidget, self.renderer, self.cutting_plane_actor, needResetCamera=False)

    def getValues(self):
        try:
            level = float(self.levelInput.text())
            angle = float(self.angleInput.text())
            
            return self.move_axis, level, angle
        except Exception as e:
            QMessageBox.warning(self, "Invalid Input", f"Inputs must be valid numbers: {e}")
            raise ValueError(f"Inputs must be valid numbers: {e}")

    def on_accept(self):
        try:
            self.emit_accepted_with_values()
            self.accept()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", f"Failed to get values: {e}")
        finally:
            self.cleanup()

    def on_reject(self):
        self.cleanup()
        self.reject()

    def cleanup(self):
        if self.cutting_plane_actor:
            remove_actor(self.vtkWidget, self.renderer, self.cutting_plane_actor, needResetCamera=False)
            self.cutting_plane_actor = None

    def closeEvent(self, event):
        self.cleanup()
        event.accept()

    def emit_accepted_with_values(self):
        try:
            axis, level, angle = self.getValues()
            self.accepted_with_values.emit(axis, level, angle)
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", f"Failed to get values: {e}")
