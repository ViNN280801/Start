from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog
from util.vtk_helpers import remove_actor, add_actor
from util.pos_file_parser import PosFileParser
from vtk import (
    vtkPoints, vtkPolyData, vtkActor, vtkLookupTable,
    vtkFloatArray, vtkArrowSource, vtkUnsignedCharArray,
    vtkGlyph3D, vtkPolyDataMapper, 
)
from .electric_field_colorbar_manager import ElectricFieldColorBarManager 


class ElectricFieldManager:
    def __init__(self, parent):
        self.results_tab = parent
        self.log_console = parent.log_console
        self.vtkWidget = parent.vtkWidget
        self.renderer = parent.renderer
        self.posFileCheckbox = parent.posFileCheckbox
        
        self.colorbar_manager = ElectricFieldColorBarManager(parent)
    
    def load_pos_file(self):
        file_dialog = QFileDialog()
        pos_file, _ = file_dialog.getOpenFileName(self.results_tab, "Open .pos file", "", "POS Files (*.pos)")

        if pos_file:
            self._process_pos_file(pos_file)
        else:
            self._uncheck_checkbox()
    
    def cleanup(self):
        self._remove_arrows_and_colorbar()
    
    def _uncheck_checkbox(self):
        self.posFileCheckbox.blockSignals(True)
        self.posFileCheckbox.setCheckState(Qt.Unchecked)
        self.posFileCheckbox.blockSignals(False)
    
    def _remove_arrows_and_colorbar(self):
        if hasattr(self, 'arrow_actor'):
            remove_actor(self.vtkWidget, self.renderer, self.arrow_actor)
            del self.arrow_actor
            
            self.colorbar_manager.cleanup()

    def _process_pos_file(self, filename):
        try:
            pos_parser = PosFileParser(filename)
            result = pos_parser.parse()
        except Exception as e:
            self._uncheck_checkbox()
            self.log_console.printError(str(e))
            return

        points_data, vectors_data = result

        points = vtkPoints()
        vectors = vtkFloatArray()
        vectors.SetNumberOfComponents(3)
        vectors.SetName("Vectors")
        vector_magnitudes = []

        for (x, y, z), (Ex, Ey, Ez) in zip(points_data, vectors_data):
            points.InsertNextPoint(x, y, z)
            magnitude = (Ex**2 + Ey**2 + Ez**2) ** 0.5
            vector_magnitudes.append(magnitude)
            vectors.InsertNextTuple3(Ex, Ey, Ez)

        self.colorbar_manager.apply_magnitudes(vector_magnitudes)

        poly_data = vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.GetPointData().SetVectors(vectors)

        lookup_table = vtkLookupTable()
        lookup_table.SetNumberOfTableValues(256)
        lookup_table.SetRange(min(vector_magnitudes), max(vector_magnitudes))
        lookup_table.Build()

        colors = vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")

        for magnitude in vector_magnitudes:
            rgb = [0.0, 0.0, 0.0]
            lookup_table.GetColor(magnitude, rgb)
            color = [int(c * 255) for c in rgb]
            colors.InsertNextTuple3(*color)

        poly_data.GetPointData().SetScalars(colors)

        arrow_source = vtkArrowSource()
        glyph = vtkGlyph3D()
        glyph.SetSourceConnection(arrow_source.GetOutputPort())
        glyph.SetInputData(poly_data)
        glyph.SetVectorModeToUseVector()
        glyph.SetScaleModeToDataScalingOff()
        glyph.SetScaleFactor(1.0)
        glyph.OrientOn()
        glyph.Update()

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        mapper.SetLookupTable(lookup_table)
        mapper.SetScalarModeToUsePointData()

        if hasattr(self, 'arrow_actor'):
            remove_actor(self.vtkWidget, self.renderer, self.arrow_actor)

        self.arrow_actor = vtkActor()
        self.arrow_actor.SetMapper(mapper)
        add_actor(self.vtkWidget, self.renderer, self.arrow_actor)

        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()
        