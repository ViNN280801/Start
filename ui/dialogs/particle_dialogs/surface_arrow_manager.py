from PyQt5.QtWidgets import QMessageBox
from vtk import (
    vtkActor,
    vtkRenderer,
    vtkPolyDataNormals,
    vtkPoints,
    vtkArrowSource,
    vtkPolyData,
    vtkGlyph3D,
    vtkFloatArray,
    vtkPolyDataMapper,
)
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from logger.log_console import LogConsole
from styles import *
from constants import *
from .particle_source_dialog import ParticleSourceDialog
from .normal_orientation_dialog import NormalOrientationDialog


class SurfaceArrowManager:
    def __init__(
        self,
        vtkWidget: QVTKRenderWindowInteractor,
        renderer: vtkRenderer,
        log_console: LogConsole,
        selected_actors: set,
        particle_source_manager,
        geditor,
    ):
        self.vtkWidget = vtkWidget
        self.renderer = renderer
        self.log_console = log_console
        self.selected_actors = selected_actors
        self.arrow_size = DEFAULT_ARROW_SCALE[0]
        self.selected_actor = None
        self.particle_source_manager = particle_source_manager
        self.geditor = geditor

    def render_editor_window(self):
        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

    def set_particle_source_as_surface(self):
        if not self.selected_actors:
            self.log_console.printWarning(
                "There is no selected surfaces to apply particle source on them"
            )
            QMessageBox.information(
                self.geditor,
                "Set Particle Source",
                "There is no selected surfaces to apply particle source on them",
            )
            return

        self.selected_actor = list(self.selected_actors)[0]
        self.select_surface_and_normals(self.selected_actor)
        if not self.selected_actor:
            return

        self.particle_source_dialog = ParticleSourceDialog(self.geditor)
        self.particle_source_dialog.accepted_signal.connect(
            lambda params: self.handle_particle_source_surface_accepted(
                params, self.data
            )
        )

    def handle_particle_source_surface_accepted(
        self, particle_params, surface_and_normals_dict
    ):
        try:
            particle_type = particle_params["particle_type"]
            energy = particle_params["energy"]
            num_particles = particle_params["num_particles"]

            self.log_console.printInfo(
                "Particle source set as surface source\n"
                f"Particle Type: {particle_type}\n"
                f"Energy: {energy} eV\n"
                f"Number of Particles: {num_particles}"
            )
            self.log_console.addNewLine()

            self.particle_source_manager.update_config_with_particle_source(
                particle_params, surface_and_normals_dict
            )
        except Exception as e:
            self.log_console.printError(f"Error setting particle source. {e}")
            QMessageBox.warning(
                self.geditor, "Particle Source", f"Error setting particle source. {e}"
            )
            return None

    def update_arrow_sizes(self, size):
        self.arrow_size = size

        # Update only the inside glyph if it's active
        if hasattr(self, "glyph_actor_inside"):
            positions_inside = self.original_positions_inside
            directions_inside = self.original_directions_inside

            self.renderer.RemoveActor(self.glyph_actor_inside)
            self.glyph_actor_inside = self.create_glyphs(
                positions_inside, directions_inside, self.arrow_size
            )
            self.renderer.AddActor(self.glyph_actor_inside)

        # Update only the outside glyph if it's active
        if hasattr(self, "glyph_actor_outside"):
            positions_outside = self.original_positions_outside
            directions_outside = self.original_directions_outside

            self.renderer.RemoveActor(self.glyph_actor_outside)
            self.glyph_actor_outside = self.create_glyphs(
                positions_outside, directions_outside, self.arrow_size
            )
            self.renderer.AddActor(self.glyph_actor_outside)

    def populate_data(self, arrows, data):
        for arrow_actor, cell_center, normal in arrows:
            actor_address = hex(id(arrow_actor))
            data[actor_address] = {"cell_center": cell_center, "normal": normal}

    def select_surface_and_normals(self, actor: vtkActor):
        poly_data = actor.GetMapper().GetInput()
        normals = self.calculate_normals(poly_data)

        if not normals:
            self.log_console.printWarning("No normals found for the selected surface")
            QMessageBox.warning(
                self.geditor,
                "Normals Calculation",
                "No normals found for the selected surface",
            )
            return

        self.num_cells = poly_data.GetNumberOfCells()
        self.positions_outside = []
        self.directions_outside = []
        self.positions_inside = []
        self.directions_inside = []
        self.data = {}

        for i in range(self.num_cells):
            normal = normals.GetTuple(i)
            rev_normal = tuple(-n for n in normal)
            cell = poly_data.GetCell(i)
            cell_center = self.calculate_cell_center(cell)

            self.positions_outside.append(cell_center)
            self.directions_outside.append(normal)
            self.positions_inside.append(cell_center)
            self.directions_inside.append(rev_normal)

            self.data[i] = {"cell_center": cell_center, "normal": normal}

        self.original_positions_outside = list(self.positions_outside)
        self.original_directions_outside = list(self.directions_outside)
        self.original_positions_inside = list(self.positions_inside)
        self.original_directions_inside = list(self.directions_inside)

        self.glyph_actor_outside = self.create_glyphs(
            self.positions_outside, self.directions_outside, self.arrow_size
        )
        self.renderer.AddActor(self.glyph_actor_outside)
        self.render_editor_window()

        self.normal_orientation_dialog = NormalOrientationDialog(
            self.arrow_size, self.geditor
        )
        self.normal_orientation_dialog.orientation_accepted.connect(
            self.handle_outside_confirmation
        )
        self.normal_orientation_dialog.size_changed.connect(self.update_arrow_sizes)
        self.normal_orientation_dialog.rejected.connect(self.cleanup)
        self.normal_orientation_dialog.show()

    def handle_outside_confirmation(self, confirmed, size):
        self.arrow_size = size
        if confirmed:
            self.finalize_surface_selection()
            self.particle_source_dialog.show()
        else:
            # Remove the outside glyph
            if hasattr(self, "glyph_actor_outside"):
                self.renderer.RemoveActor(self.glyph_actor_outside)
                del self.glyph_actor_outside

            # Recreate inside glyph
            positions = self.positions_inside
            directions = self.directions_inside

            if hasattr(self, "glyph_actor_inside"):
                self.renderer.RemoveActor(self.glyph_actor_inside)

            self.glyph_actor_inside = self.create_glyphs(
                positions, directions, self.arrow_size
            )
            self.renderer.AddActor(self.glyph_actor_inside)

            self.normal_orientation_dialog = NormalOrientationDialog(
                self.arrow_size, self.geditor
            )
            self.normal_orientation_dialog.msg_label.setText(
                "Do you want to set normals inside?"
            )
            self.normal_orientation_dialog.orientation_accepted.connect(
                self.handle_inside_confirmation
            )
            self.normal_orientation_dialog.size_changed.connect(self.update_arrow_sizes)
            self.normal_orientation_dialog.rejected.connect(self.cleanup)
            self.normal_orientation_dialog.show()

    def handle_inside_confirmation(self, confirmed, size):
        self.arrow_size = size
        if confirmed:
            positions_inside = [data["cell_center"] for data in self.data.values()]
            directions_inside = [
                (-normal[0], -normal[1], -normal[2])
                for normal in self.directions_outside
            ]

            if hasattr(self, "glyph_actor_inside"):
                self.renderer.RemoveActor(self.glyph_actor_inside)

            self.glyph_actor_inside = self.create_glyphs(
                positions_inside, directions_inside, self.arrow_size
            )
            self.renderer.AddActor(self.glyph_actor_inside)

            # Inversing of the normals to specify correct direction.
            for i, values in self.data.items():
                normal = values["normal"]
                values["normal"] = (
                    -normal[0] if normal[0] != 0 else normal[0],
                    -normal[1] if normal[1] != 0 else normal[1],
                    -normal[2] if normal[2] != 0 else normal[2],
                )

            self.finalize_surface_selection()
            self.particle_source_dialog.show()
        else:
            self.cleanup()

    def cleanup(self):
        # Remove the outside glyph if it exists
        if hasattr(self, "glyph_actor_outside"):
            self.renderer.RemoveActor(self.glyph_actor_outside)
            del self.glyph_actor_outside

        # Remove the inside glyph if it exists
        if hasattr(self, "glyph_actor_inside"):
            self.renderer.RemoveActor(self.glyph_actor_inside)
            del self.glyph_actor_inside

        self.render_editor_window()

    def finalize_surface_selection(self):
        self.cleanup()

        if not self.data:
            return

        self.log_console.printInfo(
            f"Selected surface with {self.num_cells} cells inside:"
        )
        for cell_index, values in self.data.items():
            cellCentre = values["cell_center"]
            normal = values["normal"]
            self.log_console.printInfo(
                f"Cell {cell_index}: [{cellCentre[0]:.2f}, {cellCentre[1]:.2f}, {cellCentre[2]:.2f}] - ({normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f})"
            )

        self.geditor.deselect()

    def confirm_normal_orientation(self, orientation):
        msg_box = QMessageBox(self.geditor)
        msg_box.setWindowTitle("Normal Orientation")
        msg_box.setText(f"Do you want to set normals {orientation}?")
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.No)

        result = msg_box.exec_()

        return result == QMessageBox.Yes

    def calculate_normals(self, poly_data):
        normals_filter = vtkPolyDataNormals()
        normals_filter.SetInputData(poly_data)
        normals_filter.ComputePointNormalsOff()
        normals_filter.ComputeCellNormalsOn()
        normals_filter.Update()

        return normals_filter.GetOutput().GetCellData().GetNormals()

    def calculate_cell_center(self, cell):
        cell_center = [0.0, 0.0, 0.0]
        points = cell.GetPoints()
        num_points = points.GetNumberOfPoints()
        for j in range(num_points):
            point = points.GetPoint(j)
            cell_center[0] += point[0]
            cell_center[1] += point[1]
            cell_center[2] += point[2]
        return [coord / num_points for coord in cell_center]

    def create_glyphs(self, positions, directions, arrow_size):
        arrow_source = vtkArrowSource()
        arrow_source.SetTipLength(0.2)
        arrow_source.SetShaftRadius(0.02)
        arrow_source.SetTipResolution(100)

        points = vtkPoints()
        vectors = vtkFloatArray()
        vectors.SetNumberOfComponents(3)
        vectors.SetName("Normals")

        for pos, dir in zip(positions, directions):
            points.InsertNextPoint(pos)
            vectors.InsertNextTuple(dir)

        poly_data = vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.GetPointData().SetVectors(vectors)

        glyph3D = vtkGlyph3D()
        glyph3D.SetSourceConnection(arrow_source.GetOutputPort())
        glyph3D.SetInputData(poly_data)
        glyph3D.SetVectorModeToUseVector()
        glyph3D.SetScaleModeToScaleByVector()
        glyph3D.SetScaleFactor(arrow_size)
        glyph3D.Update()

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(glyph3D.GetOutputPort())

        glyph_actor = vtkActor()
        glyph_actor.SetMapper(mapper)
        glyph_actor.GetProperty().SetColor(DEFAULT_ARROW_ACTOR_COLOR)

        return glyph_actor
