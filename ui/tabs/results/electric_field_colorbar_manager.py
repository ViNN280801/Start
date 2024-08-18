from vtk import vtkScalarBarActor, vtkLookupTable


class ElectricFieldColorBarManager:
    def __init__(self, parent):
        self.results_tab = parent
        
        self.scalarBar = vtkScalarBarActor()
        self.scalarBar.SetTitle("Electric Field, V/m")
        
        self.renderer = parent.renderer
        self.vtkWidget = parent.vtkWidget
        self.default_num_labels = 5  # Default labels count

        # Setting default style of the scalar bar
        self._setup_default_scalarbar_properties()
    
    def apply_magnitudes(self, magnitudes):
        self.min_magnitude = min(magnitudes)
        self.max_magnitude = max(magnitudes)
        self._setup_lookup_table()
        self.renderer.AddActor2D(self.scalarBar)
    
    def cleanup(self):
        self.renderer.RemoveActor2D(self.scalarBar)
        self.vtkWidget.GetRenderWindow().Render()
        
    def _setup_default_scalarbar_properties(self):
        self.scalarBar.SetWidth(0.1)
        self.scalarBar.SetHeight(0.75)
        self.scalarBar.SetDragable(True)
        self.scalarBar.SetLabelFormat("%.6e") # Count of particles is natural number
        text_property = self.scalarBar.GetLabelTextProperty()
        text_property.SetFontSize(12)
        text_property.SetFontFamilyAsString("Noto Sans SemiBold")
        text_property.SetBold(True)
        text_property.SetItalic(False)
        text_property.SetColor(0, 0, 0)

        title_text_property = self.scalarBar.GetTitleTextProperty()
        title_text_property.SetFontSize(12)
        title_text_property.SetFontFamilyAsString("Noto Sans SemiBold")
        title_text_property.SetBold(True)
        title_text_property.SetItalic(False)
        title_text_property.SetColor(0, 0, 0)
        
        # Set the position of the scalar bar (shift to the left)
        current_position = self.scalarBar.GetPosition()
        new_position_x = current_position[0] - 0.1  # Shift left
        new_position_y = current_position[1]
        self.scalarBar.SetPosition(new_position_x, new_position_y)
        
    def _setup_lookup_table(self):
        self.lookup_table = vtkLookupTable()
        self.lookup_table.SetNumberOfTableValues(256)
        self.lookup_table.SetRange(self.min_magnitude, self.max_magnitude)
        for i in range(256):
            ratio = i / 255.0
            self.lookup_table.SetTableValue(i, ratio, 0, 1 - ratio)
        self.lookup_table.Build()
        self.scalarBar.SetLookupTable(self.lookup_table)

