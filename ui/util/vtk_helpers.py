from vtk import (
    vtkUnstructuredGrid, vtkPolyData, vtkPolyDataWriter, vtkActor, vtkBooleanOperationPolyDataFilter, 
    vtkGeometryFilter, vtkPoints, vtkCellArray, vtkTriangle, vtkTransform, vtkCleanPolyData, vtkPlane,
    vtkAppendPolyData, vtkPolyDataMapper, vtkFeatureEdges, vtkPolyDataConnectivityFilter, vtkClipPolyData,
    VTK_TRIANGLE
)
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

        # Set the input objects for the operation
        operation.SetInputData(0, cleaner1.GetOutput())
        operation.SetInputData(1, cleaner2.GetOutput())

        # Update the filter to perform the subtraction
        operation.Update()

        # Retrieve the result of the subtraction
        resultPolyData = operation.GetOutput()

        # Check if subtraction was successful
        if resultPolyData is None or resultPolyData.GetNumberOfPoints() == 0:
            raise ValueError("Operation Failed: No result from the operation operation.")

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


def create_cutting_plane(axis: str, level: float):
    """
    Creates a cutting plane along a specified axis at a given level in VTK.

    Parameters:
    axis (str): The axis along which to create the cutting plane ('x', 'y', or 'z').
    level (float): The position along the specified axis where the cutting plane will be created.

    Returns:
    vtkPlane: The created cutting plane.

    Raises:
    ValueError: If the specified axis is not 'x', 'y', or 'z'.

    Examples:
    --------
    # Create a cutting plane along the z-axis at z=2.5
    cutting_plane = create_cutting_plane('z', 2.5)

    # Create a cutting plane along the x-axis at x=1.0
    cutting_plane = create_cutting_plane('x', 1.0)
    """
    plane = vtkPlane()
    plane.SetOrigin(level, 0, 0) if axis == 'x' else plane.SetOrigin(0, level, 0) if axis == 'y' else plane.SetOrigin(0, 0, level)
    
    if axis == 'x':
        plane.SetNormal(1, 0, 0)
    elif axis == 'y':
        plane.SetNormal(0, 1, 0)
    elif axis == 'z':
        plane.SetNormal(0, 0, 1)
    else:
        raise ValueError("Invalid axis")
    
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
        print(f"Error while do section on geometry: {e}")
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
