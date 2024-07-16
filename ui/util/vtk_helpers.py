from vtk import (
    vtkUnstructuredGrid, vtkPolyData, vtkPolyDataWriter, vtkActor, vtkBooleanOperationPolyDataFilter, 
    vtkGeometryFilter, vtkPoints, vtkCellArray, vtkTriangle, vtkTransform, vtkCleanPolyData, vtkPlane,
    vtkAppendPolyData, vtkPolyDataMapper, vtkFeatureEdges, vtkPolyDataConnectivityFilter, vtkClipPolyData,
    vtkRenderer,
    VTK_TRIANGLE
)
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from styles import DEFAULT_ACTOR_COLOR
from logger import InternalLogger


def convert_msh_to_vtk(msh_filename: str):
    from gmsh import open, write
    
    if not msh_filename.endswith('.msh'):
        return None

    try:        
        vtk_filename = msh_filename.replace('.msh', '.vtk')
        open(msh_filename)
        write(vtk_filename)
        return vtk_filename
    except Exception as e:
        print(f"Error converting VTK to Msh: {e}")
        return None

    
def get_polydata_from_actor(actor: vtkActor):
    mapper = actor.GetMapper()
    if hasattr(mapper, "GetInput"):
        return mapper.GetInput()
    else:
        return None


def write_vtk_polydata_to_file(polyData):
    from tempfile import NamedTemporaryFile
    
    writer = vtkPolyDataWriter()
    writer.SetInputData(polyData)

    # Create a temporary file
    temp_file = NamedTemporaryFile(delete=False, suffix='.vtk')
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
        return True  # Conversion was successful and resulted in a non-empty polyData
    else:
        return False  # Conversion failed to produce meaningful polyData


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


def convert_unstructured_grid_to_polydata(data):
    converted_part_1 = get_polydata_from_actor(data)
    converted_part_2 = convert_vtkUnstructuredGrid_to_vtkPolyData(
        converted_part_1)
    return converted_part_2


def convert_vtkPolyData_to_vtkUnstructuredGrid(polydata):
    """Convert vtkPolyData to vtkUnstructuredGrid with boundaries and surfaces."""
    boundaries = extract_boundaries(polydata)
    surfaces = extract_surfaces(polydata)

    ugrid = vtkUnstructuredGrid()
    points = vtkPoints()
    points.SetDataTypeToDouble()
    cells = vtkCellArray()

    # Copy points
    points.DeepCopy(polydata.GetPoints())

    # Copy cells
    for i in range(polydata.GetNumberOfCells()):
        cell = polydata.GetCell(i)
        cell_type = cell.GetCellType()
        ids = cell.GetPointIds()

        if cell_type == VTK_TRIANGLE:
            triangle = vtkTriangle()
            for j in range(3):
                triangle.GetPointIds().SetId(j, ids.GetId(j))
            cells.InsertNextCell(triangle)

    ugrid.SetPoints(points)
    ugrid.SetCells(VTK_TRIANGLE, cells)

    return ugrid, boundaries, surfaces


def extract_geometry_data(actor: vtkActor):
    """Extract points and cells from a vtkActor."""
    from vtkmodules.util.numpy_support import vtk_to_numpy
    
    mapper = actor.GetMapper()
    polydata = mapper.GetInput()

    ug, boundaries, surfaces = convert_vtkPolyData_to_vtkUnstructuredGrid(polydata)
    if ug is None:
        raise ValueError("Failed to convert PolyData to UnstructuredGrid")

    points = vtk_to_numpy(ug.GetPoints().GetData()).astype('float64')
    if points is None or len(points) == 0:
        raise ValueError("No points found in the UnstructuredGrid")

    cells = vtk_to_numpy(ug.GetCells().GetData())
    if cells is None or len(cells) == 0:
        raise ValueError("No cells found in the UnstructuredGrid")

    cell_offsets = vtk_to_numpy(ug.GetCellLocationsArray())
    cell_types = vtk_to_numpy(ug.GetCellTypesArray())
    cells = extract_cells(cells, cell_offsets, cell_types)
    if not cells:
        raise ValueError("Failed to extract cells")

    return points, cells, boundaries, surfaces

def extract_boundaries(polydata):
    """Extract boundaries from vtkPolyData using vtkFeatureEdges."""
    feature_edges = vtkFeatureEdges()
    feature_edges.SetInputData(polydata)
    feature_edges.BoundaryEdgesOn()
    feature_edges.FeatureEdgesOff()
    feature_edges.ManifoldEdgesOff()
    feature_edges.NonManifoldEdgesOff()
    feature_edges.Update()

    return feature_edges.GetOutput()

def extract_surfaces(polydata):
    """Extract surfaces from vtkPolyData using vtkPolyDataConnectivityFilter."""
    connectivity_filter = vtkPolyDataConnectivityFilter()
    connectivity_filter.SetInputData(polydata)
    connectivity_filter.SetExtractionModeToAllRegions()
    connectivity_filter.ColorRegionsOn()
    connectivity_filter.Update()

    return connectivity_filter.GetOutput()

def extract_cells(cells, cell_offsets, cell_types):
        """
        Helper function to extract cells in the format meshio expects.
        """
        from numpy import array
        from vtk import VTK_TRIANGLE, VTK_TETRA

        cell_dict = {}
        for offset, ctype in zip(cell_offsets, cell_types):
            if ctype in cell_dict:
                cell_dict[ctype].append(cells[offset + 1:offset + 4])
            else:
                cell_dict[ctype] = [cells[offset + 1:offset + 4]]

        meshio_cells = []
        for ctype, cell_list in cell_dict.items():
            cell_list = array(cell_list)
            if ctype == VTK_TRIANGLE:
                meshio_cells.append(("triangle", cell_list[:, :3]))
            elif ctype == VTK_TETRA:
                meshio_cells.append(("tetra", cell_list[:, :4]))

        return meshio_cells


def extract_transform_from_actor(actor: vtkActor):
    matrix = actor.GetMatrix()

    transform = vtkTransform()
    transform.SetMatrix(matrix)

    return transform

def extract_transformed_points(polydata: vtkPolyData):
    points = polydata.GetPoints()
    return [points.GetPoint(i) for i in range(points.GetNumberOfPoints())]


def get_transformation_matrix(actor: vtkActor):
    return actor.GetMatrix()


def transform_coordinates(points, matrix):
    transform = vtkTransform()
    transform.SetMatrix(matrix)
    transformed_points = []
    for point in points:
        transformed_point = transform.TransformPoint(point[0], point[1], point[2])
        transformed_points.append(transformed_point)
    return transformed_points


def compare_matrices(mat1, mat2):
    """
    Compare two vtkMatrix4x4 matrices for equality.

    Args:
        mat1 (vtkMatrix4x4): The first matrix.
        mat2 (vtkMatrix4x4): The second matrix.

    Returns:
        bool: True if the matrices are equal, False otherwise.
    """
    for i in range(4):
        for j in range(4):
            if mat1.GetElement(i, j) != mat2.GetElement(i, j):
                return False
    return True


def merge_actors(actors):
    """
    Merge the provided list of actors into a single actor.

    Args:
        actors (list): List of vtkActor objects to be merged.

    Returns:
        vtkActor: A new actor that is the result of merging the provided actors.
    """
    # Merging actors
    append_filter = vtkAppendPolyData()
    for actor in actors:
        poly_data = actor.GetMapper().GetInput()
        append_filter.AddInputData(poly_data)
    append_filter.Update()

    # Creating a new merged actor
    merged_mapper = vtkPolyDataMapper()
    merged_mapper.SetInputData(append_filter.GetOutput())

    merged_actor = vtkActor()
    merged_actor.SetMapper(merged_mapper)
    merged_actor.GetProperty().SetColor(DEFAULT_ACTOR_COLOR)

    return merged_actor

def object_operation_executor_helper(obj_from: vtkActor, obj_to: vtkActor, operation: vtkBooleanOperationPolyDataFilter):
    try:
        obj_from_subtract_polydata = convert_unstructured_grid_to_polydata(obj_from)
        obj_to_subtract_polydata = convert_unstructured_grid_to_polydata(obj_to)

        cleaner1 = vtkCleanPolyData()
        cleaner1.SetInputData(obj_from_subtract_polydata)
        cleaner1.Update()
        cleaner2 = vtkCleanPolyData()
        cleaner2.SetInputData(obj_to_subtract_polydata)
        cleaner2.Update()
        
        if cleaner1.GetOutput().GetNumberOfCells() == 0 or cleaner2.GetOutput().GetNumberOfCells() == 0:
            raise ValueError(f"Operation <{operation}> Failed: Invalid input data for the operation")

        # Set the input objects for the operation
        operation.SetInputData(0, cleaner1.GetOutput())
        operation.SetInputData(1, cleaner2.GetOutput())

        # Update the filter to perform the subtraction
        operation.Update()

        # Retrieve the result of the subtraction
        resultPolyData = operation.GetOutput()

        # Check if subtraction was successful
        if resultPolyData is None or resultPolyData.GetNumberOfPoints() == 0:
            raise ValueError(f"Operation <{operation}> Failed: No result from the operation operation.")

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(resultPolyData)

        actor = vtkActor()
        actor.SetMapper(mapper)
            
        return actor
        
    except Exception as e:
        print(InternalLogger.get_warning_none_result_with_exception_msg(e))
        return None
    

def remove_gradient(actor):
    """
    Removes gradient (scalar visibility) of the given vtkActor.

    Parameters:
    actor (vtkActor): The actor whose color needs to be set.
    """
    if actor and isinstance(actor, vtkActor):
        actor.GetMapper().ScalarVisibilityOff()


def remove_shadows(actor):
    """
    Removes shadows of the given vtkActor.

    Parameters:
    actor (vtkActor): The actor whose color needs to be set.
    """
    if actor and isinstance(actor, vtkActor):
        actor.GetProperty().SetInterpolationToFlat()


def create_cutting_plane(axis: str, level: float, angle: float = 0.0, size: float = 1e9):
    """
    Creates a cutting plane along a specified axis at a given level in VTK with optional rotation.
    
    Parameters:
    axis (str): The axis along which to create the cutting plane ('x', 'y', or 'z').
    level (float): The position along the specified axis where the cutting plane will be created.
    angle (float): The angle of rotation in degrees (default is 0, no rotation).
    size (float): The half-size of the cutting plane. The cutting plane will have dimensions (2*size, 2*size) along the other two axes.
    
    Returns:
    vtkPlane: The created cutting plane.
    
    Raises:
    ValueError: If the specified axis is not 'x', 'y', or 'z'.
    """
    plane = vtkPlane()
    transform = vtkTransform()
    
    axis = axis.lower()

    if axis == 'x':
        plane.SetOrigin(level, 0, 0)
        plane.SetNormal(1, 0, 0)
        transform.RotateWXYZ(angle, 1, 0, 0)
    elif axis == 'y':
        plane.SetOrigin(0, level, 0)
        plane.SetNormal(0, 1, 0)
        transform.RotateWXYZ(angle, 0, 1, 0)
    elif axis == 'z':
        plane.SetOrigin(0, 0, level)
        plane.SetNormal(0, 0, 1)
        transform.RotateWXYZ(angle, 0, 0, 1)
    else:
        raise ValueError(f"{InternalLogger.pretty_function_details()}: Invalid axis: {axis}")

    # Apply the rotation to the normal
    normal = plane.GetNormal()
    normal = transform.TransformNormal(normal)

    plane.SetNormal(normal)

    return plane


def cut_actor(actor: vtkActor, plane: vtkPlane):
    polydata = convert_unstructured_grid_to_polydata(actor)
    if not polydata:
        return None

    try:
        clipper1 = vtkClipPolyData()
        clipper1.SetInputData(polydata)
        clipper1.SetClipFunction(plane)
        clipper1.InsideOutOn()
        clipper1.Update()

        clipper2 = vtkClipPolyData()
        clipper2.SetInputData(polydata)
        clipper2.SetClipFunction(plane)
        clipper2.InsideOutOff()
        clipper2.Update()

        mapper1 = vtkPolyDataMapper()
        mapper1.SetInputData(clipper1.GetOutput())
        actor1 = vtkActor()
        actor1.SetMapper(mapper1)

        mapper2 = vtkPolyDataMapper()
        mapper2.SetInputData(clipper2.GetOutput())
        actor2 = vtkActor()
        actor2.SetMapper(mapper2)

        return actor1, actor2

    except Exception as e:
        print(f"Error while doing section on geometry: {e}")
        return None


def colorize_actor(actor: vtkActor, color=DEFAULT_ACTOR_COLOR):
        """
        Sets the color of the actor. If color is not provided, DEFAULT_ACTOR_COLOR is used.

        Parameters
        ----------
        actor : vtkActor
            The actor to colorize.
        color : tuple or list, optional
            The RGB color values to set. If None, DEFAULT_ACTOR_COLOR is used.
        """
        try:
            if color is None:
                color = DEFAULT_ACTOR_COLOR
            actor.GetProperty().SetColor(color)
        except Exception as e:
            print(InternalLogger.get_warning_none_result_with_exception_msg(e))
            return None


def colorize_actor_with_rgb(actor: vtkActor, r: float, g: float, b: float):
        """
        Sets the color of the actor using RGB values.

        Parameters
        ----------
        actor : vtkActor
            The actor to colorize.
        r : float
            Red component (0-1).
        g : float
            Green component (0-1).
        b : float
            Blue component (0-1).
        """
        try:
            actor.GetProperty().SetColor(r, g, b)
        except Exception as e:
            print(InternalLogger.get_warning_none_result_with_exception_msg(e))
            return None


def add_actor(vtkWidget: QVTKRenderWindowInteractor, renderer: vtkRenderer, actor: vtkActor, needResetCamera: bool = True):
    renderer.AddActor(actor)
    actor.GetProperty().SetColor(DEFAULT_ACTOR_COLOR)
    
    if needResetCamera:
        render_editor_window(vtkWidget, renderer)
    else:
        render_editor_window_without_resetting_camera(vtkWidget)


def add_actors(vtkWidget: QVTKRenderWindowInteractor, renderer: vtkRenderer, actors: list, needResetCamera: bool = True):
    for actor in actors:
        renderer.AddActor(actor)
        actor.GetProperty().SetColor(DEFAULT_ACTOR_COLOR)
    
    if needResetCamera:
        render_editor_window(vtkWidget, renderer)
    else:
        render_editor_window_without_resetting_camera(vtkWidget)


def remove_actor(vtkWidget: QVTKRenderWindowInteractor, renderer: vtkRenderer, actor: vtkActor, needResetCamera: bool = True):
    if actor and isinstance(actor, vtkActor) and actor in renderer.GetActors():
        renderer.RemoveActor(actor)
        
        if needResetCamera:
            render_editor_window(vtkWidget, renderer)
        else:
            render_editor_window_without_resetting_camera(vtkWidget)


def remove_actors(vtkWidget: QVTKRenderWindowInteractor, renderer: vtkRenderer, actors: list, needResetCamera: bool = True):
    for actor in actors:
        if actor in renderer.GetActors():
            renderer.RemoveActor(actor)
   
    if needResetCamera:
        render_editor_window(vtkWidget, renderer)
    else:
        render_editor_window_without_resetting_camera(vtkWidget)


def remove_all_actors(vtkWidget: QVTKRenderWindowInteractor, renderer: vtkRenderer, needResetCamera: bool = True):
    actors = renderer.GetActors()
    actors.InitTraversal()
    for i in range(actors.GetNumberOfItems()):
        actor = actors.GetNextActor()
        renderer.RemoveActor(actor)
    
    if needResetCamera:
        render_editor_window(vtkWidget, renderer)
    else:
        render_editor_window_without_resetting_camera(vtkWidget)


def render_editor_window(vtkWidget: QVTKRenderWindowInteractor, renderer: vtkRenderer):
    renderer.ResetCamera()
    render_editor_window_without_resetting_camera(vtkWidget)


def render_editor_window_without_resetting_camera(vtkWidget: QVTKRenderWindowInteractor):
    vtkWidget.GetRenderWindow().Render()
