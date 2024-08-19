from vtk import vtkScalarBarActor, vtkLookupTable, vtkFloatArray, vtkStringArray, vtkActor
from json import dump, load


class ParticlesColorbarManager:
    def __init__(self, parent, mesh_data=None, actor: vtkActor=None):
        self.results_tab = parent
        
        self.scalarBar = vtkScalarBarActor()
        self.mesh_data = mesh_data
        self.actor = actor
        self.renderer = parent.renderer
        self.vtkWidget = parent.vtkWidget
        self.default_num_labels = 5  # Default labels count
        self.setup_colormap()

        # Setting default style of the scalar bar
        self.setup_default_scalarbar_properties()
        
    def hide(self):
        self.scalarBar.SetVisibility(False)
    
    def show(self):
        self.scalarBar.SetVisibility(True)
        
    def setup_default_scalarbar_properties(self):
        self.scalarBar.SetWidth(0.1)
        self.scalarBar.SetHeight(0.75)
        self.scalarBar.SetDragable(True)
        self.scalarBar.SetLabelFormat("%.0f") # Count of particles is natural number
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

    def setup_colormap(self):
        self.max_count = max(triangle[5] for triangle in self.mesh_data)
        self.setup_lookup_table()
        self.set_annotations(self.default_num_labels)
        
    def setup_lookup_table(self):
        self.lookup_table = vtkLookupTable()
        self.lookup_table.SetNumberOfTableValues(256)
        self.lookup_table.SetRange(0, self.max_count)
        for i in range(256):
            ratio = i / 255.0
            self.lookup_table.SetTableValue(i, ratio, 0, 1 - ratio)
        self.lookup_table.Build()

    def set_annotations(self, num_labels=5):
        from numpy import round, linspace
        
        valuesArray = vtkFloatArray()
        annotationsArray = vtkStringArray()
        unique_values = sorted(set(triangle[5] for triangle in self.mesh_data))

        # Function to pick N evenly spaced elements including the first and the last
        def pick_n_uniformly(lst, n):
            if n >= len(lst):
                return lst
            indeces = round(linspace(0, len(lst) - 1, n)).astype(int)
            return [lst[index] for index in indeces]

        uniform_values = pick_n_uniformly(unique_values, num_labels)
        for value in uniform_values:
            valuesArray.InsertNextValue(value)
            annotationsArray.InsertNextValue(f"{value}")

        self.lookup_table.SetAnnotations(valuesArray, annotationsArray)
        
    def add_colorbar(self, title='Color Scale', range=None):
        self.scalarBar.SetTitle(title)
        self.scalarBar.SetNumberOfLabels(self.default_num_labels)

        if range:
            self.actor.GetMapper().GetLookupTable().SetRange(range[0], range[1])

        self.scalarBar.SetLookupTable(self.actor.GetMapper().GetLookupTable())
        self.renderer.AddActor2D(self.scalarBar)
        
    def apply_scale(self, width: float, height: float):
        self.scalarBar.SetWidth(width)
        self.scalarBar.SetHeight(height)
        
    def apply_color_to_actor(self, actor):
        mapper = actor.GetMapper()
        if mapper:
            mapper.SetLookupTable(self.lookup_table)
            mapper.Update()
        
    def change_font(self, font, color_rgb):
        text_property = self.scalarBar.GetLabelTextProperty()
        text_property.SetFontFamilyAsString(font.family())
        text_property.SetFontSize(font.pointSize())
        text_property.SetBold(font.bold())
        text_property.SetItalic(font.italic())
        text_property.SetColor(color_rgb)

        title_text_property = self.scalarBar.GetTitleTextProperty()
        title_text_property.SetFontFamilyAsString(font.family())
        title_text_property.SetFontSize(font.pointSize())
        title_text_property.SetBold(font.bold())
        title_text_property.SetItalic(font.italic())
        title_text_property.SetColor(color_rgb)

        self.vtkWidget.GetRenderWindow().Render()
        
    def change_divs(self, divs: int):
        self.scalarBar.SetNumberOfLabels(divs)
        self.set_annotations(divs)
        
    def reset_to_default(self):
        self.setup_default_scalarbar_properties()
        
    def get_properties(self):
        properties = {
            'position': self.scalarBar.GetPosition(),
            'width': self.scalarBar.GetWidth(),
            'height': self.scalarBar.GetHeight(),
            'num_labels': self.scalarBar.GetNumberOfLabels(),
            'label_font': {
                'family': self.scalarBar.GetLabelTextProperty().GetFontFamilyAsString(),
                'size': self.scalarBar.GetLabelTextProperty().GetFontSize(),
                'bold': self.scalarBar.GetLabelTextProperty().GetBold(),
                'italic': self.scalarBar.GetLabelTextProperty().GetItalic(),
                'color': self.scalarBar.GetLabelTextProperty().GetColor()
            },
            'title_font': {
                'family': self.scalarBar.GetTitleTextProperty().GetFontFamilyAsString(),
                'size': self.scalarBar.GetTitleTextProperty().GetFontSize(),
                'bold': self.scalarBar.GetTitleTextProperty().GetBold(),
                'italic': self.scalarBar.GetTitleTextProperty().GetItalic(),
                'color': self.scalarBar.GetTitleTextProperty().GetColor()
            }
        }
        return properties

    def save_colorbar(self, colorbar_file='scene_colorbar.json'):
        try:
            properties = self.get_properties()
            with open(colorbar_file, 'w') as f:
                dump(properties, f)
            return 1
        except Exception as e:
            print(f"Error saving colorbar: {e}")
            return None
        
    def load_colorbar(self, colorbar_file='scene_colorbar.json'):
        try:
            with open(colorbar_file, 'r') as f:
                properties = load(f)
            
            self.scalarBar.SetPosition(*properties['position'])
            self.scalarBar.SetWidth(properties['width'])
            self.scalarBar.SetHeight(properties['height'])
            self.scalarBar.SetNumberOfLabels(properties['num_labels'])
            
            label_font = properties['label_font']
            label_text_property = self.scalarBar.GetLabelTextProperty()
            label_text_property.SetFontFamilyAsString(label_font['family'])
            label_text_property.SetFontSize(label_font['size'])
            label_text_property.SetBold(label_font['bold'])
            label_text_property.SetItalic(label_font['italic'])
            label_text_property.SetColor(label_font['color'])
            
            title_font = properties['title_font']
            title_text_property = self.scalarBar.GetTitleTextProperty()
            title_text_property.SetFontFamilyAsString(title_font['family'])
            title_text_property.SetFontSize(title_font['size'])
            title_text_property.SetBold(title_font['bold'])
            title_text_property.SetItalic(title_font['italic'])
            title_text_property.SetColor(title_font['color'])
            
            self.renderer.AddActor2D(self.scalarBar)
            return 1
        except Exception as e:
            print(f"Error loading colorbar: {e}")
            return None
    
    @staticmethod
    def from_properties(vtkWidget, renderer, properties):
        actor = properties['actor']
        return ParticlesColorbarManager(vtkWidget, renderer, properties['mesh_data'], actor)
