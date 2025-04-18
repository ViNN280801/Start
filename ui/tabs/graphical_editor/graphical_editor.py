from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5.QtCore import Qt, QSize, QItemSelectionModel, pyqtSlot
from PyQt5.QtGui import QCursor, QStandardItemModel, QBrush, QIcon
from PyQt5.QtWidgets import (
    QFrame,
    QVBoxLayout,
    QHBoxLayout,
    QTreeView,
    QPushButton,
    QDialog,
    QSpacerItem,
    QColorDialog,
    QSizePolicy,
    QMessageBox,
    QFileDialog,
    QMenu,
    QAction,
    QStatusBar,
    QAbstractItemView,
)
from vtk import (
    vtkRenderer,
    vtkPolyDataMapper,
    vtkActor,
    vtkAxesActor,
    vtkOrientationMarkerWidget,
    vtkGenericDataObjectReader,
    vtkDataSetMapper,
    vtkCellPicker,
    vtkCommand,
    vtkMatrix4x4,
    vtkInteractorStyleTrackballCamera,
    vtkInteractorStyleTrackballActor,
    vtkInteractorStyleRubberBandPick,
)
from util import (
    align_view_by_axis,
    remove_last_occurrence,
    ActionHistory,
    ProjectManager,
)
from util.vtk_helpers import (
    remove_gradient,
    remove_shadows,
    render_editor_window_without_resetting_camera,
    remove_all_actors,
    compare_matrices,
    merge_actors,
    add_actor,
    add_actors,
    remove_actor,
    remove_actors,
    colorize_actor_with_rgb,
    add_gradient,
    add_shadows,
)
from util.gmsh_helpers import convert_stp_to_msh
from util.util import warning_unrealized_or_malfunctionating_function
from logger import LogConsole
from .geometry import GeometryManager
from .particle_source_manager import ParticleSourceManager
from .mesh_tree_manager import MeshTreeManager
from .geometry.geometry_constants import *
from styles import *
from constants import *
from dialogs import *
from .interactor import *


class GraphicalEditor(QFrame):
    def __init__(self, log_console: LogConsole, config_tab, parent=None):
        super().__init__(parent)

        self.config_tab = config_tab
        self.log_console = log_console
        self.mesh_file = None

        self.setup_dicts()
        self.setup_tree_view()
        self.setup_selected_actors()
        self.setup_picker(log_console)
        self.setup_toolbar()
        self.setup_ui()
        self.setup_interaction()
        self.setup_status_bar()
        self.setup_particle_source_manager()
        self.setup_axes()

        self.action_history = ActionHistory()
        self.global_undo_stack = []
        self.global_redo_stack = []

        self.isPerformOperation = (False, None)
        self.firstObjectToPerformOperation = None
        self.statusBar = QStatusBar()
        self.layout.addWidget(self.statusBar)

    def setup_dicts(self):
        # External row - is the 1st row in the tree view (volume, excluding objects like: line, point)
        # Internal row - is the 2nd row in the tree view (surface)
        # Tree dictionary (treedict) - own invented dictionary that stores data to fill the mesh tree
        self.externRow_treedict = {}  # Key = external row        |  value = treedict
        self.externRow_actors = (
            {}
        )  # Key = external row        |  value = list of actors
        self.actor_rows = (
            {}
        )  # Key = actor               |  value = pair(external row, internal row)
        self.actor_color = {}  # Key = actor               |  value = color
        self.actor_nodes = {}  # Key = actor               |  value = list of nodes
        self.actor_matrix = (
            {}
        )  # Key = actor               |  value = transformation matrix: pair(initial, current)
        self.meshfile_actors = {}  # Key = mesh filename       |  value = list of actors

    def setup_tree_view(self):
        self.treeView = QTreeView()
        self.model = QStandardItemModel()
        self.model.setHorizontalHeaderLabels(["Mesh Tree"])

    def setup_selected_actors(self):
        self.selected_actors = set()

    def setup_picker(self, log_console):
        self.picker = vtkCellPicker()
        self.picker.SetTolerance(0.005)
        self.log_console = log_console

    def setup_toolbar(self):
        self.layout = QVBoxLayout()  # Main layout
        self.toolbarLayout = QHBoxLayout()  # Layout for the toolbar

        # Create buttons for the toolbar
        self.createPointButton = self.create_button("icons/point.png", "Point")
        self.createLineButton = self.create_button("icons/line.png", "Line")
        self.createSurfaceButton = self.create_button("icons/surface.png", "Surface")
        self.createSphereButton = self.create_button("icons/sphere.png", "Sphere")
        self.createBoxButton = self.create_button("icons/box.png", "Box")
        self.createConeButton = self.create_button("icons/cone.png", "Cone")
        self.createCylinderButton = self.create_button("icons/cylinder.png", "Cylinder")
        self.uploadCustomButton = self.create_button(
            "icons/custom.png", "Upload mesh object"
        )
        self.eraseAllObjectsButton = self.create_button("icons/eraser.png", "Erase all")
        self.xAxisButton = self.create_button(
            "icons/x-axis.png", "Set camera view to X-axis"
        )
        self.yAxisButton = self.create_button(
            "icons/y-axis.png", "Set camera view to Y-axis"
        )
        self.zAxisButton = self.create_button(
            "icons/z-axis.png", "Set camera view to Z-axis"
        )
        self.centerAxisButton = self.create_button(
            "icons/center-axis.png", "Set camera view to center of axes"
        )
        self.subtractObjectsButton = self.create_button(
            "icons/subtract.png", "Subtract objects"
        )
        self.unionObjectsButton = self.create_button(
            "icons/union.png", "Combine (union) objects"
        )
        self.intersectObjectsButton = self.create_button(
            "icons/intersection.png", "Intersection of two objects"
        )
        self.crossSectionButton = self.create_button(
            "icons/cross-section.png", "Cross section of the object"
        )
        self.setBoundaryConditionsSurfaceButton = self.create_button(
            "icons/boundary-conditions-surface.png",
            "Turning on mode to select boundary nodes on surface",
        )
        self.setParticleSourceButton = self.create_button(
            "icons/particle-source.png", "Set particle source as surface"
        )
        self.meshObjectsButton = self.create_button(
            "icons/mesh-objects.png",
            "Mesh created objects. WARNING: After this action list of the created objects will be zeroed up",
        )

        self.spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.toolbarLayout.addSpacerItem(self.spacer)

        # Connect buttons to methods
        self.createPointButton.clicked.connect(self.create_point)
        self.createLineButton.clicked.connect(self.create_line)
        self.createSurfaceButton.clicked.connect(self.create_surface)
        self.createSphereButton.clicked.connect(self.create_sphere)
        self.createBoxButton.clicked.connect(self.create_box)
        self.createConeButton.clicked.connect(self.create_cone)
        self.createCylinderButton.clicked.connect(self.create_cylinder)
        self.uploadCustomButton.clicked.connect(self.upload_custom)
        self.eraseAllObjectsButton.clicked.connect(self.clear_scene_and_tree_view)
        self.xAxisButton.clicked.connect(lambda: self.align_view_by_axis("x"))
        self.yAxisButton.clicked.connect(lambda: self.align_view_by_axis("y"))
        self.zAxisButton.clicked.connect(lambda: self.align_view_by_axis("z"))
        self.centerAxisButton.clicked.connect(lambda: self.align_view_by_axis("center"))
        self.subtractObjectsButton.clicked.connect(self.subtract_button_clicked)
        self.unionObjectsButton.clicked.connect(self.combine_button_clicked)
        self.intersectObjectsButton.clicked.connect(self.intersection_button_clicked)
        self.crossSectionButton.clicked.connect(self.cross_section_button_clicked)
        self.setBoundaryConditionsSurfaceButton.clicked.connect(
            self.activate_selection_boundary_conditions_mode_for_surface
        )
        self.setParticleSourceButton.clicked.connect(self.set_particle_source)
        self.meshObjectsButton.clicked.connect(self.mesh_objects)

    def setup_ui(self):
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.layout.addLayout(self.toolbarLayout)
        self.layout.addWidget(self.vtkWidget)
        self.setLayout(self.layout)

        self.renderer = vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)

        self.treeView.setContextMenuPolicy(Qt.CustomContextMenu)
        self.treeView.customContextMenuRequested.connect(self.on_treeView_context_menu)

    def setup_interaction(self):
        self.change_interactor(INTERACTOR_STYLE_TRACKBALL_CAMERA)
        self.interactor.AddObserver(vtkCommand.KeyPressEvent, self.on_key_press)

    def setup_status_bar(self):
        self.statusBar = QStatusBar()
        self.layout.addWidget(self.statusBar)

    def setup_particle_source_manager(self):
        self.particle_source_manager = ParticleSourceManager(
            self.vtkWidget,
            self.renderer,
            self.log_console,
            self.config_tab,
            self.selected_actors,
            self.statusBar,
            self,
        )

    def setup_axes(self):
        self.axes_actor = vtkAxesActor()
        self.axes_widget = vtkOrientationMarkerWidget()
        self.axes_widget.SetOrientationMarker(self.axes_actor)
        self.axes_widget.SetInteractor(self.vtkWidget.GetRenderWindow().GetInteractor())
        self.axes_widget.SetViewport(0.0, 0.0, 0.2, 0.2)
        self.axes_widget.EnabledOn()
        self.axes_widget.InteractiveOff()

    def get_tree_dict_by_extern_row(self, extern_row):
        return self.externRow_treedict.get(extern_row, None)

    def get_extern_row_by_treedict(self, treedict):
        for extern_row, td in self.externRow_treedict.items():
            if td == treedict:
                return extern_row
        return None

    def get_actors_by_extern_row(self, extern_row):
        return self.externRow_actors.get(extern_row, [])

    def get_extern_row_by_actor(self, actor):
        for extern_row, actors in self.externRow_actors.items():
            if actor in actors:
                return extern_row
        return None

    def get_rows_by_actor(self, actor):
        return self.actor_rows.get(actor, None)

    def get_actors_by_extern_row_from_actorRows(self, extern_row):
        return [
            actor
            for actor, (ext_row, _) in self.actor_rows.items()
            if ext_row == extern_row
        ]

    def get_color_by_actor(self, actor):
        return self.actor_color.get(actor, None)

    def actor_color_add(self, actor, color):
        self.actor_color[actor] = color

    def get_actors_by_color(self, color):
        return [actor for actor, clr in self.actor_color.items() if clr == color]

    def get_nodes_by_actor(self, actor):
        return self.actor_nodes.get(actor, [])

    def get_actors_by_node(self, node):
        return [actor for actor, nodes in self.actor_nodes.items() if node in nodes]

    def get_matrix_by_actor(self, actor):
        return self.actor_matrix.get(actor, None)

    def get_actors_by_matrix(self, matrix):
        return [
            actor for actor, matrices in self.actor_matrix.items() if matrices == matrix
        ]

    def get_actors_by_filename(self, filename):
        return self.meshfile_actors.get(filename, [])

    def get_filename_by_actor(self, actor):
        for filename, actors in self.meshfile_actors.items():
            if actor in actors:
                return filename
        return None

    def get_filenames_from_dict(self) -> list:
        filenames = []
        for filename, actors in self.meshfile_actors.items():
            filenames.append(filename)
        return filenames

    def get_index_from_rows(self, external_row, internal_row):
        # Get the external index
        external_index = self.treeView.model().index(external_row, 0)
        if not external_index.isValid():
            return None

        # Get the internal index using the external index as parent
        internal_index = self.treeView.model().index(internal_row, 0, external_index)
        if not internal_index.isValid():
            return None

        return internal_index

    def update_actor_dictionaries(
        self, actor_to_add: vtkActor, volume_row: int, surface_row: int, filename: str
    ):
        self.actor_rows[actor_to_add] = (volume_row, surface_row)
        self.actor_color[actor_to_add] = DEFAULT_ACTOR_COLOR
        self.actor_matrix[actor_to_add] = (
            actor_to_add.GetMatrix(),
            actor_to_add.GetMatrix(),
        )
        self.meshfile_actors.setdefault(filename, []).append(actor_to_add)

    def update_actor_dictionaries(self, actor_to_remove: vtkActor, actor_to_add=None):
        """
        Remove actor_to_remove from all dictionaries and add actor_to_add to those dictionaries if provided.

        Args:
            actor_to_remove (vtkActor): The actor to remove from all dictionaries.
            actor_to_add (vtkActor, optional): The actor to add to all dictionaries. Defaults to None.
        """
        if actor_to_remove in self.actor_rows:
            volume_row, surface_row = self.actor_rows[actor_to_remove]

            # Remove actor from actor_rows
            del self.actor_rows[actor_to_remove]

            # Remove actor from actor_color
            if actor_to_remove in self.actor_color:
                del self.actor_color[actor_to_remove]

            # Remove actor from actor_matrix
            if actor_to_remove in self.actor_matrix:
                del self.actor_matrix[actor_to_remove]

            # Remove actor from meshfile_actors
            for filename, actors in self.meshfile_actors.items():
                if actor_to_remove in actors:
                    actors.remove(actor_to_remove)
                    break

            # If an actor to add is provided, add it to the dictionaries
            if actor_to_add:
                self.actor_rows[actor_to_add] = (volume_row, surface_row)
                self.actor_color[actor_to_add] = DEFAULT_ACTOR_COLOR
                self.actor_matrix[actor_to_add] = (
                    actor_to_add.GetMatrix(),
                    actor_to_add.GetMatrix(),
                )
                self.meshfile_actors.setdefault(filename, []).append(actor_to_add)

    @pyqtSlot()
    def activate_selection_boundary_conditions_mode_slot(self):
        self.setBoundaryConditionsSurfaceButton.click()

    def initialize_tree(self):
        self.model = QStandardItemModel()
        self.model.setHorizontalHeaderLabels(["Mesh Tree"])
        self.setTreeViewModel()

    def setTreeViewModel(self):
        self.treeView.setModel(self.model)
        self.treeView.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.treeView.selectionModel().selectionChanged.connect(
            self.on_tree_selection_changed
        )

    def upload_mesh_file(self, file_path):
        from os.path import exists, isfile

        if exists(file_path) and isfile(file_path):
            self.clear_scene_and_tree_view()
            self.mesh_file = file_path
            self.initialize_tree()
            treedict = MeshTreeManager.get_tree_dict(self.mesh_file)
            self.add_actors_and_populate_tree_view(treedict, file_path)
        else:
            QMessageBox.warning(self, "Warning", f"Unable to open file {file_path}")
            return None

    def erase_all_from_tree_view(self):
        self.model.clear()
        self.model.setHorizontalHeaderLabels(["Mesh Tree"])

    def get_actor_from_mesh(self, mesh_file_path: str):
        vtk_file_path = mesh_file_path.replace(".msh", ".vtk")

        reader = vtkGenericDataObjectReader()
        reader.SetFileName(vtk_file_path)
        reader.Update()

        if reader.IsFilePolyData():
            mapper = vtkPolyDataMapper()
        elif reader.IsFileUnstructuredGrid():
            mapper = vtkDataSetMapper()
        else:
            return
        mapper.SetInputConnection(reader.GetOutputPort())

        actor = vtkActor()
        actor.SetMapper(mapper)
        add_actor(self.vtkWidget, self.renderer, actor)

        return actor

    def create_button(self, icon_path, tooltip, size=(40, 40)):
        button = QPushButton()
        button.setIcon(QIcon(icon_path))
        button.setIconSize(QSize(*size))
        button.setFixedSize(QSize(*size))
        button.setToolTip(tooltip)
        self.toolbarLayout.addWidget(button)
        return button

    def on_treeView_context_menu(self, position):
        indexes = self.treeView.selectedIndexes()
        if not indexes:
            return

        menu = QMenu()

        move_action = QAction("Move", self)
        rotate_action = QAction("Rotate", self)
        adjust_size_action = QAction("Scale", self)
        remove_action = QAction("Remove", self)
        colorize_action = QAction("Colorize", self)
        remove_gradient_action = QAction("Remove gradient (scalar visibility)", self)
        remove_shadows_action = QAction("Remove shadows", self)
        merge_surfaces_action = QAction("Merge surfaces", self)
        add_material_action = QAction("Add Material", self)
        hide_action = QAction("Hide", self)

        move_action.triggered.connect(self.move_actors)
        rotate_action.triggered.connect(self.rotate_actors)
        adjust_size_action.triggered.connect(self.scale_actors)
        remove_action.triggered.connect(self.permanently_remove_actors)
        colorize_action.triggered.connect(self.colorize_actors)
        remove_gradient_action.triggered.connect(self.remove_gradient)
        remove_shadows_action.triggered.connect(self.remove_shadows)
        merge_surfaces_action.triggered.connect(self.merge_surfaces)
        add_material_action.triggered.connect(self.add_material)
        hide_action.triggered.connect(self.hide_actors)

        # Determine if all selected actors are visible or not
        all_visible = all(
            actor.GetVisibility()
            for actor in self.selected_actors
            if actor and isinstance(actor, vtkActor)
        )
        hide_action.setText("Hide" if all_visible else "Show")

        menu.addAction(move_action)
        menu.addAction(rotate_action)
        menu.addAction(adjust_size_action)
        menu.addAction(remove_action)
        menu.addAction(colorize_action)
        menu.addAction(remove_gradient_action)
        menu.addAction(remove_shadows_action)
        menu.addAction(merge_surfaces_action)
        menu.addAction(add_material_action)
        menu.addAction(hide_action)

        menu.exec_(self.treeView.viewport().mapToGlobal(position))

    def hide_actors(self):
        action = self.sender()
        all_visible = all(
            actor.GetVisibility()
            for actor in self.selected_actors
            if actor and isinstance(actor, vtkActor)
        )

        for actor in self.selected_actors:
            if actor and isinstance(actor, vtkActor):
                cur_visibility = actor.GetVisibility()
                actor.SetVisibility(not cur_visibility)
                self.update_tree_item_visibility(actor, not cur_visibility)
                self.add_action(ACTION_ACTOR_HIDE, actor, not cur_visibility)

        if all_visible:
            action.setText("Show")
        else:
            action.setText("Hide")

        render_editor_window_without_resetting_camera(self.vtkWidget)

    def update_tree_item_visibility(self, actor, visible):
        rows = self.get_rows_by_actor(actor)
        if rows:
            external_row, internal_row = rows
            index = self.get_index_from_rows(external_row, internal_row)
            if index.isValid():
                item = self.treeView.model().itemFromIndex(index)
                if item:
                    original_name = item.text()
                    if visible:
                        if original_name.endswith(" (hidden)"):
                            # Remove " (hidden)"
                            item.setText(original_name[:-9])
                        item.setForeground(QBrush(DEFAULT_TREE_VIEW_ROW_COLOR))
                    else:
                        if not original_name.endswith(" (hidden)"):
                            # Add " (hidden)"
                            item.setText(original_name + " (hidden)")
                        item.setForeground(
                            QBrush(DEFAULT_TREE_VIEW_ROW_COLOR_HIDED_ACTOR)
                        )

    def rename_tree_item(self, rows, new_name):
        """
        Rename a tree view item identified by the provided rows.

        :param rows: A tuple containing the external and internal rows.
        :param new_name: The new name to set for the tree view item.
        """
        external_row, internal_row = rows
        index = self.get_index_from_rows(external_row, internal_row)
        if index.isValid():
            item = self.treeView.model().itemFromIndex(index)
            if item:
                item.setText(new_name)

    def get_filename_from_dialog(self) -> str:
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getSaveFileName(
            None,
            "Save Mesh File",
            "",
            "Mesh Files (*.msh);;All Files (*)",
            options=options,
        )

        if not filename:
            return None
        if not filename.endswith(".msh"):
            filename += ".msh"

        return filename

    def create_point(self):
        dialog = PointDialog(self)
        if dialog.exec_() == QDialog.Accepted and dialog.getValues() is not None:
            try:
                point = dialog.getPoint()
                point_actor = GeometryManager.create_point(point)

                if point_actor:
                    add_actor(self.vtkWidget, self.renderer, point_actor)

                    point_dimtags = GeometryManager.get_dimtags_by_actor(point_actor)
                    self.add_action(
                        ACTION_ACTOR_CREATING, "point", point_actor, point_dimtags
                    )

            except Exception as e:
                QMessageBox.warning(self, "Create Point", str(e))

    def create_line(self):
        dialog = LineDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            try:
                line = dialog.getLine()
                line_actor = GeometryManager.create_line(line)

                if line_actor:
                    add_actor(self.vtkWidget, self.renderer, line_actor)

                    line_dimtags = GeometryManager.get_dimtags_by_actor(line_actor)
                    self.add_action(
                        ACTION_ACTOR_CREATING, "line", line_actor, line_dimtags
                    )

            except Exception as e:
                QMessageBox.warning(self, "Create Line", str(e))

    def create_surface(self):
        dialog = SurfaceDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            try:
                surface = dialog.getSurface()
                if not surface:
                    return

                surface_actor = GeometryManager.create_surface(surface)

                if surface_actor:
                    add_actor(self.vtkWidget, self.renderer, surface_actor)

                    surface_dimtags = GeometryManager.get_dimtags_by_actor(
                        surface_actor
                    )
                    self.add_action(
                        ACTION_ACTOR_CREATING, "surface", surface_actor, surface_dimtags
                    )

            except Exception as e:
                QMessageBox.warning(self, "Create Surface", str(e))

    def create_sphere(self):
        dialog = SphereDialog(self)
        if dialog.exec_() == QDialog.Accepted and dialog.getValues() is not None:
            try:
                sphere = dialog.getSphere()
                sphere_actor = GeometryManager.create_sphere(sphere)

                if sphere_actor:
                    add_actor(self.vtkWidget, self.renderer, sphere_actor)

                    sphere_dimtags = GeometryManager.get_dimtags_by_actor(sphere_actor)
                    self.add_action(
                        ACTION_ACTOR_CREATING, "sphere", sphere_actor, sphere_dimtags
                    )

            except Exception as e:
                QMessageBox.warning(self, "Create Sphere", str(e))

    def create_box(self):
        dialog = BoxDialog(self)
        if dialog.exec_() == QDialog.Accepted and dialog.getValues() is not None:
            try:
                box = dialog.getBox()
                box_actor = GeometryManager.create_box(box)

                if box_actor:
                    add_actor(self.vtkWidget, self.renderer, box_actor)

                    box_dimtags = GeometryManager.get_dimtags_by_actor(box_actor)
                    self.add_action(
                        ACTION_ACTOR_CREATING, "box", box_actor, box_dimtags
                    )

            except Exception as e:
                QMessageBox.warning(self, "Create Box", str(e))

    def create_cone(self):
        dialog = ConeDialog(self)
        if dialog.exec_() == QDialog.Accepted and dialog.getValues() is not None:
            try:
                cone = dialog.getCone()
                cone_actor = GeometryManager.create_cone(cone)

                if cone_actor:
                    add_actor(self.vtkWidget, self.renderer, cone_actor)

                    cone_dimtags = GeometryManager.get_dimtags_by_actor(cone_actor)
                    self.add_action(
                        ACTION_ACTOR_CREATING, "cone", cone_actor, cone_dimtags
                    )

            except Exception as e:
                QMessageBox.warning(self, "Create Cone", str(e))

    def create_cylinder(self):
        dialog = CylinderDialog(self)
        if dialog.exec_() == QDialog.Accepted and dialog.getValues() is not None:
            try:
                warning_unrealized_or_malfunctionating_function("Creating cylinder")
                cylinder = dialog.getCylinder()
                cylinder_actor = GeometryManager.create_cylinder(cylinder)

                if cylinder_actor:
                    add_actor(self.vtkWidget, self.renderer, cylinder_actor)

                    cylinder_dimtags = GeometryManager.get_dimtags_by_actor(
                        cylinder_actor
                    )
                    self.add_action(
                        ACTION_ACTOR_CREATING,
                        "cylinder",
                        cylinder_actor,
                        cylinder_dimtags,
                    )

            except Exception as e:
                QMessageBox.warning(self, "Create Cylinder", str(e))

    def upload_custom(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Mesh or Geometry File",
            "",
            "Mesh Files (*.msh);;Step Files (*.stp);;VTK Files (*.vtk);;All Files (*)",
            options=options,
        )

        if not file_name:
            return

        # If the selected file is a STEP file, prompt for conversion parameters.
        if file_name.endswith(".stp"):
            dialog = MeshDialog(self)
            if dialog.exec() == QDialog.Accepted:
                mesh_size, mesh_dim = dialog.getValues()
                try:
                    converted_file_name = convert_stp_to_msh(
                        file_name, mesh_size, mesh_dim
                    )
                    if converted_file_name:
                        self.add_custom(converted_file_name)
                except ValueError as e:
                    QMessageBox.warning(self, "Invalid Input", str(e))
            else:
                QMessageBox.critical(
                    self,
                    "Error",
                    "Dialog was closed by user. Invalid mesh size or mesh dimensions.",
                )
        elif file_name.endswith(".msh") or file_name.endswith(".vtk"):
            self.add_custom(file_name)
            self.log_console.printInfo(
                f"Successfully uploaded custom object from {file_name}"
            )

    def permanently_remove_actors(self):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Warning)
        msgBox.setText(
            "Are you sure you want to delete the object? It will be permanently deleted."
        )
        msgBox.setWindowTitle("Permanently Object Deletion")
        msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

        choice = msgBox.exec()
        if choice == QMessageBox.No:
            return
        else:
            for actor in self.selected_actors:
                if actor and isinstance(actor, vtkActor):
                    row = self.get_volume_row(actor)
                    if row is None:
                        remove_actor(self.vtkWidget, self.renderer, actor)
                        GeometryManager.remove(
                            self.vtkWidget, self.renderer, actor, needResetCamera=True
                        )
                        return

                    actors = self.get_actor_from_volume_row(row)
                    if not actors:
                        self.log_console.printInternalError(
                            f"Can't find actors <{hex(id(actors))}> by tree view row [{row}]>"
                        )
                        return

                    self.remove_row_from_tree(row)
                    remove_actors(self.vtkWidget, self.renderer, actors)
                    self.action_history.remove_by_id(self.action_history.get_id())
                    self.action_history.decrementIndex()

    def colorize_actors(self):
        actorColor = QColorDialog.getColor()
        if actorColor.isValid():
            r, g, b = actorColor.redF(), actorColor.greenF(), actorColor.blueF()

            for actor in self.selected_actors:
                if actor and isinstance(actor, vtkActor):
                    colorize_actor_with_rgb(actor, r, g, b)
                    color = actor.GetProperty().GetColor()
                    self.actor_color_add(actor, color)
                    self.add_action(ACTION_ACTOR_COLORIZE, actor, color)
            self.deselect()

    def remove_gradient(self):
        for actor in self.selected_actors:
            if actor and isinstance(actor, vtkActor):
                remove_gradient(actor)
                self.add_action(ACTION_ACTOR_GRADIENT, actor)
        self.deselect()

    def remove_shadows(self):
        for actor in self.selected_actors:
            if actor and isinstance(actor, vtkActor):
                remove_shadows(actor)
                self.add_action(ACTION_ACTOR_SHADOW, actor)
        self.deselect()

    def add_custom(self, meshfilename: str):
        customTreeDict = MeshTreeManager.get_tree_dict(meshfilename)
        self.add_actors_and_populate_tree_view(customTreeDict, meshfilename, "volume")

    def global_undo(self):
        if not self.global_undo_stack:
            return
        action = self.global_undo_stack.pop()
        self.global_redo_stack.append(action)

        if action == ACTION_ACTOR_ADDING:
            self.undo_adding()
        elif action == ACTION_ACTOR_CREATING:
            self.undo_creating()
        elif action == ACTION_ACTOR_TRANSFORMATION:
            self.undo_transform()
        elif action == ACTION_ACTOR_HIDE:
            self.undo_hide()
        elif action == ACTION_ACTOR_COLORIZE:
            self.undo_colorize()
        elif action == ACTION_ACTOR_GRADIENT:
            self.undo_gradient()
        elif action == ACTION_ACTOR_SHADOW:
            self.undo_shadow()
        # TODO: Make other undo/redo functionality

    def global_redo(self):
        if not self.global_redo_stack:
            return
        action = self.global_redo_stack.pop()
        self.global_undo_stack.append(action)

        if action == ACTION_ACTOR_ADDING:
            self.redo_adding()
        elif action == ACTION_ACTOR_CREATING:
            self.redo_creating()
        elif action == ACTION_ACTOR_TRANSFORMATION:
            self.redo_transform()
        elif action == ACTION_ACTOR_HIDE:
            self.redo_hide()
        elif action == ACTION_ACTOR_COLORIZE:
            self.redo_colorize()
        elif action == ACTION_ACTOR_GRADIENT:
            self.redo_gradient()
        elif action == ACTION_ACTOR_SHADOW:
            self.redo_shadow()

    def global_undo_stack_remove(self, action: str):
        remove_last_occurrence(self.global_undo_stack, action)

    def global_redo_stack_remove(self, action: str):
        remove_last_occurrence(self.global_redo_stack, action)

    def undo_transform(self):
        res = self.action_history.undo()
        if not res or len(res) != 5:
            return

        action_name, actor, x_value, y_value, z_value = res
        GeometryManager.transform_general(
            action_name, actor, -x_value, -y_value, -z_value
        )

        self.global_redo_stack.append(ACTION_ACTOR_TRANSFORMATION)
        self.global_undo_stack_remove(ACTION_ACTOR_TRANSFORMATION)

    def redo_transform(self):
        res = self.action_history.redo()
        if not res or len(res) != 5:
            return

        action_name, actor, x_value, y_value, z_value = res
        GeometryManager.transform_general(action_name, actor, x_value, y_value, z_value)

        self.global_undo_stack.append(ACTION_ACTOR_TRANSFORMATION)
        self.global_redo_stack_remove(ACTION_ACTOR_TRANSFORMATION)

    def undo_adding(self):
        res = self.action_history.undo()
        if not res or len(res) != 4:
            return
        row, actors, treedict, objType = res

        remove_actors(self.vtkWidget, self.renderer, actors)

        if objType != "line":
            self.remove_row_from_tree(row)
        else:
            self.remove_rows_from_tree(row)

        self.global_redo_stack.append(ACTION_ACTOR_ADDING)
        self.global_undo_stack_remove(ACTION_ACTOR_ADDING)

    def redo_adding(self):
        res = self.action_history.redo()
        if not res or len(res) != 4:
            return
        row, actors, treedict, objType = res

        add_actors(self.vtkWidget, self.renderer, actors)
        self.populate_tree(treedict, objType)

        self.global_undo_stack.append(ACTION_ACTOR_ADDING)
        self.global_redo_stack_remove(ACTION_ACTOR_ADDING)

    def undo_creating(self):
        res = self.action_history.undo()
        if not res or len(res) != 3:
            return
        obj_str, actor, dimtags = res

        try:
            remove_actor(self.vtkWidget, self.renderer, actor)
            GeometryManager.delete_by_actor(actor)
        except Exception as e:
            self.log_console.printInternalError(
                f"Failed to delete actor from the GeometryManager: {e}"
            )
            return

        self.global_redo_stack.append(ACTION_ACTOR_CREATING)
        self.global_undo_stack_remove(ACTION_ACTOR_CREATING)

    def redo_creating(self):
        res = self.action_history.redo()
        if not res or len(res) != 3:
            return
        obj_str, actor, dimtags = res

        add_actor(self.vtkWidget, self.renderer, actor)
        GeometryManager.add(obj_str, actor, dimtags)

        self.global_undo_stack.append(ACTION_ACTOR_CREATING)
        self.global_redo_stack_remove(ACTION_ACTOR_CREATING)

    def undo_hide(self):
        res = self.action_history.undo()
        if not res or len(res) != 2:
            return

        actor, visibility = res
        actor.SetVisibility(visibility)
        self.update_tree_item_visibility(actor, visibility)

        self.global_redo_stack.append(ACTION_ACTOR_HIDE)
        self.global_undo_stack_remove(ACTION_ACTOR_HIDE)

    def redo_hide(self):
        res = self.action_history.redo()
        if not res or len(res) != 2:
            return

        actor, visibility = res
        actor.SetVisibility(not visibility)
        self.update_tree_item_visibility(actor, not visibility)

        self.global_undo_stack.append(ACTION_ACTOR_HIDE)
        self.global_redo_stack_remove(ACTION_ACTOR_HIDE)

    def undo_colorize(self):
        res = self.action_history.undo()
        if not res or len(res) != 2:
            return

        actor, color = res
        actor.GetProperty().SetColor(color)
        del self.actor_color[actor]

        self.global_redo_stack.append(ACTION_ACTOR_COLORIZE)
        self.global_undo_stack_remove(ACTION_ACTOR_COLORIZE)

    def redo_colorize(self):
        res = self.action_history.redo()
        if not res or len(res) != 2:
            return

        actor, color = res
        actor.GetProperty().SetColor(color)
        self.actor_color_add(actor, color)

        self.global_undo_stack.append(ACTION_ACTOR_COLORIZE)
        self.global_redo_stack_remove(ACTION_ACTOR_COLORIZE)

    def undo_gradient(self):
        res = self.action_history.undo()
        if not res or len(res) != 1:
            return

        actor = res
        add_gradient(actor)

        self.global_redo_stack.append(ACTION_ACTOR_GRADIENT)
        self.global_undo_stack_remove(ACTION_ACTOR_GRADIENT)

    def redo_gradient(self):
        res = self.action_history.redo()
        if not res or len(res) != 1:
            return

        actor = res
        remove_gradient(actor)

        self.global_undo_stack.append(ACTION_ACTOR_GRADIENT)
        self.global_redo_stack_remove(ACTION_ACTOR_GRADIENT)

    def undo_shadow(self):
        res = self.action_history.undo()
        if not res or len(res) != 1:
            return

        actor = res
        add_shadows(actor)

        self.global_redo_stack.append(ACTION_ACTOR_SHADOW)
        self.global_undo_stack_remove(ACTION_ACTOR_SHADOW)

    def redo_shadow(self):
        res = self.action_history.redo()
        if not res or len(res) != 1:
            return

        actor = res
        remove_shadows(actor)

        self.global_undo_stack.append(ACTION_ACTOR_SHADOW)
        self.global_redo_stack_remove(ACTION_ACTOR_SHADOW)

    def remove_row_from_tree(self, row):
        self.model.removeRow(row)

    def remove_rows_from_tree(self, rows):
        row = rows[0]
        for _ in range(len(rows)):
            self.model.removeRow(row)

    def get_transformed_actors(self):
        """
        Identify all actors that have been transformed and update the actor_matrix.

        Returns:
            list: A list of transformed actors along with their filenames.
        """
        transformed_actors = set()

        for actor, (initial_transform, _) in self.actor_matrix.items():
            current_transform = actor.GetMatrix()

            if not compare_matrices(current_transform, initial_transform):
                filename = None
                for fn, actors in self.meshfile_actors.items():
                    if actor in actors:
                        filename = fn
                        break
                transformed_actors.add((actor, filename))

                # Update the actor_matrix with the new transform
                new_transform = vtkMatrix4x4()
                new_transform.DeepCopy(current_transform)
                self.actor_matrix[actor] = (initial_transform, new_transform)

        return transformed_actors

    def fill_dicts(self, row, actors, objType: str, filename: str):
        """
        Populate the new dictionaries with actors and their initial transformations.

        Args:
            row (int): The row index in the tree view.
            actors (list): List of vtkActor objects.
            objType (str): The type of object ('volume', 'line', etc.).
            filename (str): The mesh filename associated with the actors.
        """
        if objType == "volume":
            volume_row = row
            for i, actor in enumerate(actors):
                if actor and isinstance(actor, vtkActor):
                    surface_row = volume_row + i
                    actor_color = actor.GetProperty().GetColor()
                    initial_transform = vtkMatrix4x4()
                    initial_transform.DeepCopy(actor.GetMatrix())

                    self.actor_rows[actor] = (volume_row, surface_row)
                    self.actor_color[actor] = actor_color
                    self.actor_matrix[actor] = (initial_transform, initial_transform)
                    if filename in self.meshfile_actors:
                        self.meshfile_actors[filename].append(actor)
                    else:
                        self.meshfile_actors[filename] = [actor]

        elif objType == "line":
            for i, r in enumerate(row):
                actor = actors[i]
                if actor and isinstance(actor, vtkActor):
                    actor_color = actor.GetProperty().GetColor()
                    initial_transform = vtkMatrix4x4()
                    initial_transform.DeepCopy(actor.GetMatrix())

                    self.actor_rows[actor] = (r, r)
                    self.actor_color[actor] = actor_color
                    self.actor_matrix[actor] = (initial_transform, initial_transform)
                    if filename in self.meshfile_actors:
                        self.meshfile_actors[filename].append(actor)
                    else:
                        self.meshfile_actors[filename] = [actor]

        else:
            for actor in actors:
                if actor and isinstance(actor, vtkActor):
                    actor_color = actor.GetProperty().GetColor()
                    initial_transform = vtkMatrix4x4()
                    initial_transform.DeepCopy(actor.GetMatrix())

                    self.actor_rows[actor] = (row, row)
                    self.actor_color[actor] = actor_color
                    self.actor_matrix[actor] = (initial_transform, initial_transform)
                    if filename in self.meshfile_actors:
                        self.meshfile_actors[filename].append(actor)
                    else:
                        self.meshfile_actors[filename] = [actor]

    def get_volume_row(self, actor):
        """
        Get the volume index for the given actor.

        Args:
            actor (vtkActor): The actor for which to get the volume index.

        Returns:
            int: The volume index, or None if the actor is not found.
        """
        if actor in self.actor_rows:
            return self.actor_rows[actor][0]
        return None

    def get_surface_row(self, actor):
        """
        Get the surface index for the given actor.

        Args:
            actor (vtkActor): The actor for which to get the surface index.

        Returns:
            int: The surface index, or None if the actor is not found.
        """
        if actor in self.actor_rows:
            return self.actor_rows[actor][1]
        return None

    def get_actor_from_volume_row(self, volume_row):
        """
        Get the list of actors for the given volume index.

        Args:
            volume_row (int): The volume index for which to get the actors.

        Returns:
            list: A list of actors for the given volume index, or None if not found.
        """
        actors = [
            actor
            for actor, (vol_idx, _) in self.actor_rows.items()
            if vol_idx == volume_row
        ]
        return actors if actors else None

    def get_actor_from_surface_row(self, surface_row):
        """
        Get the actor for the given surface index.

        Args:
            surface_row (int): The surface index for which to get the actor.

        Returns:
            vtkActor: The actor for the given surface index, or None if not found.
        """
        for actor, (_, surf_idx) in self.actor_rows.items():
            if surf_idx == surface_row:
                return actor
        return None

    def add_actor(self, actor):
        add_actor(self.vtkWidget, self.renderer, actor)

    def remove_actor(self, actor):
        remove_actor(self.vtkWidget, self.renderer, actor)

    def fill_actor_nodes(self, treedict: dict, objType: str):
        # Ensure the dictionary exists
        if not hasattr(self, "actor_nodes"):
            self.actor_nodes = {}

        # Update the actor_nodes with the new data
        self.actor_nodes.update(
            MeshTreeManager.form_actor_nodes_dictionary(
                treedict, self.actor_rows, objType
            )
        )

    def populate_tree(self, treedict: dict, objType: str, filename: str) -> list:
        row = MeshTreeManager.populate_tree_view(
            treedict, self.action_history._id, self.model, self.treeView, objType
        )
        self.treeView.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.treeView.selectionModel().selectionChanged.connect(
            self.on_tree_selection_changed
        )
        actors = MeshTreeManager.create_actors_from_tree_dict(treedict, objType)

        self.fill_dicts(row, actors, objType, filename)
        self.fill_actor_nodes(treedict, objType)

        return row, actors

    def add_actors_and_populate_tree_view(
        self, treedict: dict, filename: str, objType: str = "volume"
    ):
        self.action_history.incrementIndex()
        row, actors = self.populate_tree(treedict, objType, filename)
        add_actors(self.vtkWidget, self.renderer, actors)

        self.externRow_treedict[row] = treedict
        if row not in self.externRow_actors:
            self.externRow_actors[row] = []
        self.externRow_actors[row].append(actors)

        self.add_action(ACTION_ACTOR_ADDING, row, actors, treedict, objType)

    def add_action(self, action: str, *args):
        if action == ACTION_ACTOR_ADDING:
            if (
                len(args) == 4
                and isinstance(args[0], int)
                and isinstance(args[1], list)
                and isinstance(args[2], dict)
                and isinstance(args[3], str)
            ):
                row, actors, treedict, objType = args
                self.action_history.add_action((row, actors, treedict, objType))
                self.action_history.incrementIndex()
                self.global_undo_stack.append(ACTION_ACTOR_ADDING)
            else:
                raise ValueError("Invalid arguments for ACTION_ACTOR_ADDING")

        elif action == ACTION_ACTOR_CREATING:
            if (
                len(args) == 3
                and isinstance(args[0], str)
                and isinstance(args[1], vtkActor)
                and isinstance(args[2], list)
            ):
                objType, actor, dimtags = args
                self.action_history.add_action((objType, actor, dimtags))
                self.action_history.incrementIndex()
                self.global_undo_stack.append(ACTION_ACTOR_CREATING)
            else:
                raise ValueError("Invalid arguments for ACTION_ACTOR_CREATING")

        elif action == ACTION_ACTOR_TRANSFORMATION:
            if (
                len(args) == 5
                and isinstance(args[0], str)
                and isinstance(args[1], vtkActor)
                and isinstance(args[2], float)
                and isinstance(args[3], float)
                and isinstance(args[4], float)
            ):
                trandform_name, actor, x_value, y_value, z_value = args
                self.action_history.add_action(
                    (trandform_name, actor, x_value, y_value, z_value)
                )
                self.action_history.incrementIndex()
                self.global_undo_stack.append(ACTION_ACTOR_TRANSFORMATION)
            else:
                raise ValueError("Invalid arguments for ACTION_ACTOR_TRANSFORMATION")

        elif action == ACTION_ACTOR_HIDE:
            if (
                len(args) == 2
                and isinstance(args[0], vtkActor)
                and isinstance(args[1], bool)
            ):
                actor, visibility = args
                self.action_history.add_action((actor, visibility))
                self.action_history.incrementIndex()
                self.global_undo_stack.append(ACTION_ACTOR_HIDE)
            else:
                raise ValueError("Invalid arguments for ACTION_ACTOR_HIDE")

        elif action == ACTION_ACTOR_COLORIZE:
            if len(args) == 2 and isinstance(args[0], vtkActor):
                actor, color = args
                self.action_history.add_action((actor, color))
                self.action_history.incrementIndex()
                self.global_undo_stack.append(ACTION_ACTOR_COLORIZE)
            else:
                raise ValueError("Invalid arguments for ACTION_ACTOR_COLORIZE")

        elif action == ACTION_ACTOR_GRADIENT:
            if len(args) == 1 and isinstance(args[0], vtkActor):
                actor = args
                self.action_history.add_action((actor))
                self.action_history.incrementIndex()
                self.global_undo_stack.append(ACTION_ACTOR_GRADIENT)
            else:
                raise ValueError("Invalid arguments for ACTION_ACTOR_GRADIENT")

        elif action == ACTION_ACTOR_SHADOW:
            if len(args) == 1 and isinstance(args[0], vtkActor):
                actor = args
                self.action_history.add_action((actor))
                self.action_history.incrementIndex()
                self.global_undo_stack.append(ACTION_ACTOR_SHADOW)
            else:
                raise ValueError("Invalid arguments for ACTION_ACTOR_SHADOW")

        else:
            raise ValueError(f"Unknown action type {action}")

    def restore_actor_colors(self):
        try:
            for actor, color in self.actor_color.items():
                actor.GetProperty().SetColor(color)
            render_editor_window_without_resetting_camera(self.vtkWidget)
        except Exception as e:
            self.log_console.printError(f"Error in restore_actor_colors: {e}")

    def highlight_actors(self, actors):
        for actor in actors:
            actor.GetProperty().SetColor(DEFAULT_SELECTED_ACTOR_COLOR)
        render_editor_window_without_resetting_camera(self.vtkWidget)

    def unhighlight_actors(self):
        self.restore_actor_colors()

    def on_tree_selection_changed(self):
        selected_indexes = self.treeView.selectedIndexes()
        if not selected_indexes:
            return

        self.unhighlight_actors()
        self.selected_actors.clear()

        for index in selected_indexes:
            selected_row = index.row()
            parent_index = index.parent().row()

            selected_item = None
            if parent_index == -1:
                for actor, (volume_row, _) in self.actor_rows.items():
                    if volume_row == selected_row:
                        selected_item = actor
                        break
            else:
                for actor, (volume_row, surface_row) in self.actor_rows.items():
                    if volume_row == parent_index and surface_row == selected_row:
                        selected_item = actor
                        break

            if selected_item:
                self.highlight_actors([selected_item])
                self.selected_actors.add(selected_item)

    def retrieve_mesh_filename(self) -> str:
        """
        Retrieve the mesh filename from the configuration tab.

        If a mesh file is specified in the configuration tab, use it as the filename.
        Otherwise, use the default temporary mesh file.

        Returns:
        str: The mesh filename to be used.
        """
        mesh_filename = DEFAULT_TEMP_MESH_FILE
        if self.config_tab.mesh_file:
            mesh_filename = self.config_tab.mesh_file
        return mesh_filename

    def pick_actor(self, obj, event):
        click_pos = self.interactor.GetEventPosition()
        self.picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)

        actor = self.picker.GetActor()
        if actor:
            if not (self.interactor.GetControlKey() or self.interactor.GetShiftKey()):
                # Reset selection of all previous actors and tree view items
                self.unhighlight_actors()
                self.reset_selection_treeview()
                self.selected_actors.clear()

            rows_to_select = ()

            # Find the corresponding rows
            for act, (volume_row, surface_row) in self.actor_rows.items():
                if actor == act:
                    rows_to_select = (volume_row, surface_row)
                    break

            # Select the rows in the tree view if rows_to_select is not empty
            if rows_to_select:
                index = self.model.index(
                    rows_to_select[1], 0, self.model.index(rows_to_select[0], 0)
                )
                self.treeView.selectionModel().select(
                    index, QItemSelectionModel.Select | QItemSelectionModel.Rows
                )

            actor.GetProperty().SetColor(DEFAULT_SELECTED_ACTOR_COLOR)
            self.selected_actors.add(actor)
            render_editor_window_without_resetting_camera(self.vtkWidget)

        # Call the original OnLeftButtonDown event handler to maintain default interaction behavior
        self.interactorStyle.OnLeftButtonDown()

        if self.isPerformOperation[0]:
            operationDescription = self.isPerformOperation[1]

            if not self.firstObjectToPerformOperation:
                self.firstObjectToPerformOperation = list(self.selected_actors)[0]
                self.statusBar.showMessage(
                    f"With which object to perform {operationDescription}?"
                )
            else:
                secondObjectToPerformOperation = list(self.selected_actors)[0]
                if (
                    self.firstObjectToPerformOperation
                    and secondObjectToPerformOperation
                ):
                    operationType = self.isPerformOperation[1]

                    if (
                        self.firstObjectToPerformOperation
                        == secondObjectToPerformOperation
                    ):
                        QMessageBox.warning(
                            self,
                            f"{operationType}",
                            f"Can't perform operation {operationType} on itself",
                        )
                        self.firstObjectToPerformOperation = None
                        secondObjectToPerformOperation = None
                        self.isPerformOperation = (False, None)
                        self.statusBar.clearMessage()
                        self.deselect()
                        return

                    if operationType == "subtract":
                        self.subtract_objects(
                            self.firstObjectToPerformOperation,
                            secondObjectToPerformOperation,
                        )
                    elif operationType == "union":
                        self.combine_objects(
                            self.firstObjectToPerformOperation,
                            secondObjectToPerformOperation,
                        )
                    elif operationType == "intersection":
                        self.intersect_objects(
                            self.firstObjectToPerformOperation,
                            secondObjectToPerformOperation,
                        )
                else:
                    QMessageBox.warning(
                        self,
                        "Warning",
                        "No objects have been selected for the operation.",
                    )

                self.firstObjectToPerformOperation = None
                self.isPerformOperation = (False, None)
                self.statusBar.clearMessage()

    def on_left_button_press(self, obj, event):
        self.pick_actor(obj, event)

    def on_right_button_press(self, obj, event):
        click_pos = self.interactor.GetEventPosition()
        self.picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)

        actor = self.picker.GetActor()
        if actor:
            self.selected_actors.add(actor)
            self.context_menu()

    def on_key_press(self, obj, event):
        key = self.interactor.GetKeySym()

        if key == "Escape":
            self.change_interactor(INTERACTOR_STYLE_TRACKBALL_CAMERA)
            self.deselect()

        if key == "Delete" or key == "BackSpace":
            if self.selected_actors:
                self.remove_objects_with_restore(self.selected_actors)

        # C - controlling the object.
        if key == "c" or key == "C":
            if self.selected_actors:
                self.change_interactor(INTERACTOR_STYLE_TRACKBALL_ACTOR)

        self.interactorStyle.OnKeyPress()

    def context_menu(self):
        menu = QMenu(self)

        move_action = QAction("Move", self)
        move_action.triggered.connect(self.move_actors)
        menu.addAction(move_action)

        change_angle_action = QAction("Rotate", self)
        change_angle_action.triggered.connect(self.rotate_actors)
        menu.addAction(change_angle_action)

        adjust_size_action = QAction("Adjust size", self)
        adjust_size_action.triggered.connect(self.scale_actors)
        menu.addAction(adjust_size_action)

        remove_object_action = QAction("Remove", self)
        remove_object_action.triggered.connect(self.permanently_remove_actors)
        menu.addAction(remove_object_action)

        colorize_object_action = QAction("Colorize", self)
        colorize_object_action.triggered.connect(self.colorize_actors)
        menu.addAction(colorize_object_action)

        remove_gradient_action = QAction("Remove gradient (scalar visibility)", self)
        remove_gradient_action.triggered.connect(self.remove_gradient)
        menu.addAction(remove_gradient_action)

        remove_shadows_action = QAction("Remove shadows", self)
        remove_shadows_action.triggered.connect(self.remove_shadows)
        menu.addAction(remove_shadows_action)

        merge_surfaces_action = QAction("Merge surfaces", self)
        merge_surfaces_action.triggered.connect(self.merge_surfaces)
        menu.addAction(merge_surfaces_action)

        add_material_action = QAction("Add Material", self)
        add_material_action.triggered.connect(self.add_material)
        menu.addAction(add_material_action)

        hide_action = QAction("Hide", self)
        hide_action.triggered.connect(self.hide_actors)
        menu.addAction(hide_action)

        menu.exec_(QCursor.pos())

    def reset_selection_treeview(self):
        self.treeView.clearSelection()

    def set_color(self, actor: vtkActor, color):
        try:
            actor.GetProperty().SetColor(color)
        except:
            self.log_console.printInternalError(
                f"Can't set color [{color}] to actor <{hex(id(actor))}>"
            )
            return

    def set_particle_source(self):
        self.particle_source_manager.set_particle_source()

    def reset_particle_source_arrow(self):
        self.particle_source_manager.reset_particle_source_arrow()

    def deselect(self):
        try:
            for actor in self.renderer.GetActors():
                if actor in self.actor_color:
                    original_color = self.actor_color[actor]
                else:
                    original_color = DEFAULT_ACTOR_COLOR
                actor.GetProperty().SetColor(original_color)

            self.selected_actors.clear()
            self.vtkWidget.GetRenderWindow().Render()
            self.reset_selection_treeview()

        except Exception as e:
            self.log_console.printError(f"Error in deselect: {e}")

    def move_actors(self):
        dialog = MoveActorDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            x_offset, y_offset, z_offset = dialog.getValues()

            for actor in self.selected_actors:
                GeometryManager.move(actor, x_offset, y_offset, z_offset)
                self.add_action(
                    ACTION_ACTOR_TRANSFORMATION,
                    "move",
                    actor,
                    x_offset,
                    y_offset,
                    z_offset,
                )

            self.deselect()

    def rotate_actors(self):
        dialog = AngleDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            angle_x, angle_y, angle_z = dialog.getValues()

            for actor in self.selected_actors:
                GeometryManager.rotate(actor, angle_x, angle_y, angle_z)
                self.add_action(
                    ACTION_ACTOR_TRANSFORMATION,
                    "rotate",
                    actor,
                    angle_x,
                    angle_y,
                    angle_z,
                )

            self.deselect()

    def scale_actors(self):
        try:
            dialog = ScaleDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                x_scale, y_scale, z_scale = dialog.getValues()

                for actor in self.selected_actors:
                    GeometryManager.scale(actor, x_scale, y_scale, z_scale)
                    self.add_action(
                        ACTION_ACTOR_TRANSFORMATION,
                        "scale",
                        actor,
                        x_scale,
                        y_scale,
                        z_scale,
                    )

                self.deselect()
            warning_unrealized_or_malfunctionating_function(f"Adjusting actor {actor}")
        except Exception as e:
            QMessageBox.warning(self, "Scale Object", str(e))

    def change_interactor(self, style: str):
        self.interactor = self.vtkWidget.GetRenderWindow().GetInteractor()

        if style == INTERACTOR_STYLE_TRACKBALL_CAMERA:
            self.interactorStyle = vtkInteractorStyleTrackballCamera()
            self.picker = vtkCellPicker()  # Use single object picker
            self.interactorStyle.AddObserver(
                vtkCommand.LeftButtonPressEvent, self.on_left_button_press
            )
            self.interactorStyle.AddObserver(
                vtkCommand.RightButtonPressEvent, self.on_right_button_press
            )
        elif style == INTERACTOR_STYLE_TRACKBALL_ACTOR:
            self.interactorStyle = vtkInteractorStyleTrackballActor()
            self.picker = vtkCellPicker()  # Use single object picker
            self.log_console.printWarning(
                "Interactor style changed: Be careful with arbitrary object transformation! If you want to set boundary conditions for this object, they will apply to the old coordinates of the nodes. Because the program does not provide for changes to key objects for which boundary conditions are set"
            )
        elif style == INTERACTOR_STYLE_RUBBER_AND_PICK:
            self.interactorStyle = vtkInteractorStyleRubberBandPick()
            self.interactorStyle.AddObserver(
                vtkCommand.LeftButtonPressEvent, self.on_left_button_press
            )
        else:
            QMessageBox.warning(
                self,
                "Change Interactor",
                f"Can't change current interactor style. There is no such interactor: {style}",
            )
            self.log_console.printWarning(
                f"Can't change current interactor style. There is no such interactor: {style}"
            )

        self.interactor.SetInteractorStyle(self.interactorStyle)
        self.interactor.Initialize()
        self.interactor.Start()

    def extract_indices(self, actors):
        """
        Extract surface indices and volume row for the provided list of actors.

        Args:
            actors (list): List of vtkActor objects.

        Returns:
            tuple: A tuple containing the list of surface indices and the volume row.
        """
        surface_indices = []
        volume_row = None
        for actor in actors:
            surface_row = self.get_surface_row(actor)
            if surface_row is not None:
                surface_indices.append(surface_row)
                if volume_row is None:
                    volume_row = self.get_volume_row(actor)
        return volume_row, surface_indices

    def merge_surfaces(self):
        if len(self.selected_actors) == 1:
            self.log_console.printInfo("Nothing to merge, selected only 1 surface")
            QMessageBox.warning(
                self, "Merge Surfaces", "Nothing to merge, selected only 1 surface"
            )
            return

        # Extracting indices for the actors to be merged
        volume_row, surface_indices = self.extract_indices(self.selected_actors)
        if not surface_indices or volume_row is None:
            self.log_console.printError(
                "No valid surface indices found for selected actors"
            )
            return

        volume_ids = set()
        for actor in self.selected_actors:
            volume_id, surface_id = self.actor_rows.get(actor, (None, None))
            if volume_id is not None:
                volume_ids.add(volume_id)

        if len(volume_ids) > 1:
            self.log_console.printError("Selected surfaces belong to different volumes")
            QMessageBox.warning(
                self,
                "Merge Surfaces",
                "Application doesn't support merging between surfaces from different volumes",
            )
            return

        # Remove selected actors and save the first one for future use
        saved_actor = vtkActor()
        for actor in self.selected_actors:
            self.update_actor_dictionaries(actor)
            saved_actor = actor
        remove_actors(self.vtkWidget, self.renderer, self.selected_actors)
        merged_actor = merge_actors(self.selected_actors)

        # Adding the merged actor to the scene
        add_actor(self.vtkWidget, self.renderer, merged_actor)
        self.update_actor_dictionaries(saved_actor, merged_actor)
        self.update_tree_view(volume_row, surface_indices, merged_actor)

        self.log_console.printInfo(
            f"Successfully merged selected surfaces: {set([i + 1 for i in surface_indices])} to object with id <{hex(id(merged_actor))}>"
        )
        self.deselect()

    def update_tree_view(self, volume_row, surface_indices, merged_actor):
        model = self.treeView.model()
        MeshTreeManager.update_tree_view(model, volume_row, surface_indices)

        # Updating the actor_rows dictionary with the new internal row index
        self.actor_rows[merged_actor] = (volume_row, surface_indices[0])

        # Adjusting indices in actor_rows after removal
        self.adjust_actor_rows(volume_row, surface_indices)

    def adjust_actor_rows(self, volume_row, removed_indices):
        """
        Adjust the actor_rows dictionary to maintain sequential numbering of surface indices after merging.

        Args:
            volume_row (int): The row index of the volume item in the tree view.
            removed_indices (list): The list of indices that were removed.
        """
        # Sort removed indices for proper processing
        removed_indices = sorted(removed_indices)

        # Initialize a list of current surface indices
        current_surface_indices = [
            rows[1] for actor, rows in self.actor_rows.items() if rows[0] == volume_row
        ]
        current_surface_indices.sort()

        # Create a new list of sequential indices
        new_indices = []
        new_index = 0
        for i in range(len(current_surface_indices) + len(removed_indices)):
            if i not in removed_indices:
                new_indices.append(new_index)
                new_index += 1

        # Update actor_rows with new sequential indices
        index_mapping = dict(zip(current_surface_indices, new_indices))
        for actor, (vol_row, surf_row) in list(self.actor_rows.items()):
            if vol_row == volume_row and surf_row in index_mapping:
                self.actor_rows[actor] = (vol_row, index_mapping[surf_row])

    def align_view_by_axis(self, axis: str):
        align_view_by_axis(axis, self.renderer, self.vtkWidget)

    def save_scene(
        self,
        logConsole,
        fontColor,
        actors_file="scene_actors_meshTab.vtk",
        camera_file="scene_camera_meshTab.json",
    ):
        ProjectManager.save_scene(
            self.renderer,
            logConsole,
            fontColor,
            actors_file=actors_file,
            camera_file=camera_file,
        )

    def load_scene(
        self,
        logConsole,
        fontColor,
        actors_file="scene_actors_meshTab.vtk",
        camera_file="scene_camera_meshTab.json",
    ):
        ProjectManager.load_scene(
            self.vtkWidget,
            self.renderer,
            logConsole,
            fontColor,
            actors_file=actors_file,
            camera_file=camera_file,
        )

    def get_total_count_of_actors(self):
        return self.renderer.GetActors().GetNumberOfItems()

    def clear_scene_and_tree_view(self):
        # There is no need to ask about assurance of deleting when len(actors) = 0
        if self.get_total_count_of_actors() == 0:
            return

        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Warning)
        msgBox.setWindowTitle("Deleting All The Data")
        msgBox.setText(
            "Are you sure you want to delete all the objects? They will be permanently deleted."
        )
        msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

        choice = msgBox.exec()
        if choice == QMessageBox.Yes:
            self.erase_all_from_tree_view()  # 1. Erasing all records from the tree view
            remove_all_actors(
                self.vtkWidget, self.renderer
            )  # 2. Deleting all the actors
            GeometryManager.clear()
        self.action_history.clearIndex()

    def subtract_button_clicked(self):
        self.deselect()
        self.isPerformOperation = (True, "subtract")
        self.statusBar.showMessage("From which obejct subtract?")
        self.operationType = "subtraction"

    def combine_button_clicked(self):
        self.deselect()
        self.isPerformOperation = (True, "union")
        self.statusBar.showMessage("What object to combine?")
        self.operationType = "union"

    def intersection_button_clicked(self):
        self.deselect()
        self.isPerformOperation = (True, "intersection")
        self.statusBar.showMessage("What object to intersect?")
        self.operationType = "intersection"

    def cross_section_button_clicked(self):
        if not self.selected_actors:
            QMessageBox.warning(
                self, "Cross Section", "You need to select object first"
            )
            return

        if len(self.selected_actors) != 1:
            QMessageBox.warning(
                self, "Cross Section", "Can't perform cross section on several objects"
            )
            return

        dialog = TwoPointSelectionDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            point1, point2 = dialog.getPoints()
        else:
            return

        try:
            dialog = AxisSelectionDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                axis = dialog.getSelectedAxis()

                actor_to_cut = list(self.selected_actors)[0]
                out_actor1, out_actor2 = GeometryManager.cross_section(
                    actor_to_cut, point1, point2, axis
                )

                remove_actor(
                    self.vtkWidget, self.renderer, actor_to_cut, needResetCamera=False
                )
                add_actors(
                    self.vtkWidget,
                    self.renderer,
                    [out_actor1, out_actor2],
                    needResetCamera=True,
                )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to perform plane cut: {e}")
            self.log_console.printError(f"Failed to perform plane cut: {e}")

    def subtract_objects(self, obj_from: vtkActor, obj_to: vtkActor):
        result_actor = GeometryManager.subtract(obj_from, obj_to)
        self.replace_actors_with_result(obj_from, obj_to, result_actor)

    def combine_objects(self, obj_from: vtkActor, obj_to: vtkActor):
        result_actor = GeometryManager.combine(obj_from, obj_to)
        self.replace_actors_with_result(obj_from, obj_to, result_actor)

    def intersect_objects(self, obj_from: vtkActor, obj_to: vtkActor):
        result_actor = GeometryManager.intersect(obj_from, obj_to)
        self.replace_actors_with_result(obj_from, obj_to, result_actor)

    def replace_actors_with_result(
        self, first: vtkActor, second: vtkActor, result: vtkActor
    ):
        remove_actor(self.vtkWidget, self.renderer, first)
        remove_actor(self.vtkWidget, self.renderer, second)
        add_actor(self.vtkWidget, self.renderer, result)

    def save_boundary_conditions(self, node_ids, value):
        from json import dump, load, JSONDecodeError

        try:
            with open(self.config_tab.config_file_path, "r") as file:
                data = load(file)
        except FileNotFoundError:
            data = {}
        except JSONDecodeError as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Error parsing JSON file '{self.config_tab.config_file_path}': {e}",
            )
            return
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred while reading the configuration file '{self.config_tab.config_file_path}': {e}",
            )
            return

        if "Boundary Conditions" not in data:
            data["Boundary Conditions"] = {}

        node_key = ",".join(map(str, node_ids))
        data["Boundary Conditions"][node_key] = value

        try:
            with open(self.config_tab.config_file_path, "w") as file:
                dump(data, file, indent=4)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save configuration: {e}")

    def activate_selection_boundary_conditions_mode_for_surface(self):
        if not self.selected_actors:
            QMessageBox.information(
                self,
                "Set Boundary Conditions",
                "There is no selected surfaces to apply boundary conditions on them",
            )
            return

        dialog = BoundaryValueInputDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            value, ok = dialog.get_value()
            if not ok:
                QMessageBox.warning(
                    self,
                    "Set Boundary Conditions Value",
                    "Failed to apply value, retry please",
                )
                return
        else:
            return

        for actor in self.selected_actors:
            if actor in self.actor_nodes:
                nodes = self.actor_nodes[actor]
                self.save_boundary_conditions(nodes, value)
                self.log_console.printInfo(
                    f"Object: {hex(id(actor))}, Nodes: {nodes}, Value: {value}"
                )
        self.deselect()

    def mesh_objects(self):
        if GeometryManager.empty():
            QMessageBox.information(
                self, "Mesh Simple Objects", "There is no objects to mesh"
            )
            self.log_console.printInfo("There is no objects to mesh")
            return

        mesh_filename = self.get_filename_from_dialog()
        if mesh_filename:
            dialog = mesh_dialog.MeshDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                mesh_size, mesh_dim = dialog.getValues()
                success = GeometryManager.mesh(mesh_filename, mesh_dim, mesh_size)

                if not success:
                    self.log_console.printWarning(
                        "Something went wrong while saving and meshing created objects"
                    )
                    self.clear_scene_and_tree_view()
                    return
                else:
                    self.log_console.printInfo(
                        "Deleting objects from the list of the created objects..."
                    )

    def add_material(self):
        dialog = AddMaterialDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            selected_material = dialog.materials_combobox.currentText()
            if not selected_material:
                QMessageBox.warning(
                    self,
                    "Add Material",
                    "Can't add material, you need to assign name to the material first",
                )
                return
            # TODO: Handle the selected material here (e.g., add it to the application)
            pass
