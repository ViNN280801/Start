from gmsh import initialize, finalize, isInitialized, clear, model, option
from logger.internal_logger import InternalLogger
from math import radians
from sys import stderr


def gmsh_init():
    """Initialize Gmsh if it is not already initialized."""
    if not isInitialized():
        initialize()


def gmsh_finalize():
    """Finalize Gmsh if it is initialized."""
    if isInitialized():
        finalize()
    
        
def gmsh_clear():
    """
    Clear all loaded models and post-processing data
    """
    clear()


def convert_stp_to_msh(filename: str, mesh_size: float, mesh_dim: int):
    from gmsh import write
    
    try:
        check_stp_filename(filename)
        check_mesh_size(mesh_size)
        check_mesh_dim(mesh_dim)
        
        model.occ.importShapes(filename)
        model.occ.synchronize()
        option.setNumber("Mesh.MeshSizeMin", mesh_size)
        option.setNumber("Mesh.MeshSizeMax", mesh_size)

        output_file = filename.replace(".stp", ".msh")
        write(output_file)
    except Exception as e:
        raise RuntimeError(f"An error occurred during conversion: {e}")
    finally:
        return output_file


def check_msh_filename(filename: str):
    """
    Check if the filename ends with '.msh' and if the path is accessible.

    Args:
    filename (str): The filename to check.

    Raises:
    TypeError: If the filename is not a string.
    ValueError: If the filename does not end with '.msh' extension.
    OSError: If the path is not accessible or writable.
    """
    if not isinstance(filename, str):
        raise TypeError("Filename must be a string.")
    
    if not filename.endswith(".msh"):
        raise ValueError("Filename must end with '.msh' extension.")

    from util import check_path_access
    check_path_access(filename)


def check_stp_filename(filename: str):
    """
    Check if the filename ends with '.stp' and if the path is accessible.

    Args:
    filename (str): The filename to check.

    Raises:
    TypeError: If the filename is not a string.
    ValueError: If the filename does not end with '.stp' extension.
    OSError: If the path is not accessible or writable.
    """
    if not isinstance(filename, str):
        raise TypeError("Filename must be a string.")
    
    if not filename.endswith(".stp"):
        raise ValueError("Filename must end with '.stp' extension.")

    from util import check_path_access
    check_path_access(filename)


def check_mesh_dim(mesh_dim: int):
    """
    Check if the mesh dimension is within the bounds [0; 3].

    Args:
    mesh_dim (int): The mesh dimension to check.

    Raises:
    TypeError: If the mesh dimension is not an integer.
    ValueError: If the mesh dimension is not within the bounds [0; 3].
    """
    if not isinstance(mesh_dim, int):
        raise TypeError("Mesh dimension must be an integer.")
    
    if not 0 <= mesh_dim <= 3:
        raise ValueError("Mesh dimension must be within the bounds [0; 3].")


def check_mesh_size(mesh_size: float):
    """
    Check if the mesh size is a positive float.

    Args:
    mesh_size (float): The mesh size to check.

    Raises:
    TypeError: If the mesh size is not a float.
    ValueError: If the mesh size is not positive.
    """
    if not isinstance(mesh_size, float):
        raise TypeError("Mesh size must be a float.")
    
    if mesh_size <= 0:
        raise ValueError("Mesh size must be positive.")


def check_tag(tag: int):
    """
    Checks the provided tag value.

    If the tag value is -1, raises a ValueError indicating that the tag is invalid.
    Otherwise, the function does nothing.

    Parameters:
    tag (int): The tag value to check.

    Raises:
    ValueError: If the tag value is -1.
    """
    if tag == -1:
        raise ValueError("Invalid GMSH tag: -1.")


def check_dimtags(dimtags):
    """
    Checks the dimensions of the provided dimtags.

    Iterates through the provided dimtags and ensures that all dimensions are equal to 3.
    If any dimension is not equal to 3, raises a ValueError indicating the invalid dimension.

    Parameters:
    dimtags (list of tuples): A list of (dimension, tag) tuples to check.

    Raises:
    ValueError: If any dimension in the dimtags is not equal to 3.
    """
    if not dimtags:
        raise ValueError(f"{InternalLogger.pretty_function_details()}: Invalid dimtags: {dimtags}")
    
    for dim, _ in dimtags:
        if dim != 3:
            raise ValueError(f"Invalid dimension {dim} in dimtags. Only dimension 3 is supported.")


def complete_dimtag(geometry: str, tag: int):
    """
    Compose dimtag from one tag and geometry string representation.
    
    Args:
    geometry (str): The string representation of the geometry.
    'point':    0,
    'line':     1,
    'surface':  2,
    'sphere':   3,
    'box':      3,
    'cone':     3,
    'cylinder': 3
    
    tag (int): The tag to be used for the geometry.
    
    Returns:
    list: A list containing a single tuple (dimension, tag).
    """
    geometry = geometry.lower()
    dim = {
        'point': 0,
        'line': 1,
        'surface': 2,
        'sphere': 3,
        'box': 3,
        'cone': 3,
        'cylinder': 3
    }.get(geometry, None)
    
    if dim is None:
        raise ValueError(f"Unsupported geometry type: {geometry}")
    
    return [(dim, tag)]


def complete_dimtags(geometry: str, tags: list):
    """
    Compose dimtags from multiple tags and geometry string representation.
    
    Args:
    geometry (str): The string representation of the geometry.
    'point':    0,
    'line':     1,
    'surface':  2,
    'sphere':   3,
    'box':      3,
    'cone':     3,
    'cylinder': 3
    
    tags (list): A list of tags to be used for the geometry.
    
    Returns:
    list: A list containing tuples (dimension, tag) for each tag.
    """
    geometry = geometry.lower()
    dim = {
        'point': 0,
        'line': 1,
        'surface': 2,
        'sphere': 3,
        'box': 3,
        'cone': 3,
        'cylinder': 3
    }.get(geometry, None)
    
    if dim is None:
        raise ValueError(f"Unsupported geometry type: {geometry}")
    
    return [(dim, tag) for tag in tags]


def gmsh_rotate(dimtags, angle_x: float, angle_y: float, angle_z: float):
    """
    Rotates the geometries identified by dimtags by the given angles around the x, y, and z axes.

    Parameters:
    dimtags (list of tuples): A list of (dimension, tag) tuples identifying the geometries to rotate.
    angle_x (float): The angle to rotate around the x-axis, in radians.
    angle_y (float): The angle to rotate around the y-axis, in radians.
    angle_z (float): The angle to rotate around the z-axis, in radians.

    Raises:
    ValueError: If any dimension in the dimtags is not equal to 3.
    """
    try:
        check_dimtags(dimtags)
        origin = [0, 0, 0]
           
        if angle_x != 0:
            model.occ.rotate(dimtags, origin[0], origin[1], origin[2], origin[0] + 1, origin[1], origin[2], angle_x)
        if angle_y != 0:
            model.occ.rotate(dimtags, origin[0], origin[1], origin[2], origin[0], origin[1] + 1, origin[2], angle_y)
        if angle_z != 0:
            model.occ.rotate(dimtags, origin[0], origin[1], origin[2], origin[0], origin[1], origin[2] + 1, angle_z)
            
        model.occ.synchronize()
        
    except Exception as e:
        print(f"Error rotating geometry with GMSH: {e}", file=stderr)


def create_cutting_plane(axis: str, level: float, angle: float = 0.0, size: float = 1e9):
    """
    Creates a cutting plane along a specified axis at a given level.

    Parameters:
    axis (str): The axis along which to create the cutting plane ('x', 'y', or 'z').
    level (float): The position along the specified axis where the cutting plane will be created.
    angle (float): The angle of rotation in degrees (default is 0, no rotation).
    size (float): The half-size of the cutting plane. The cutting plane will have dimensions (2*size, 2*size) along the other two axes.

    Returns:
    int: The tag of the created cutting plane.

    Raises:
    ValueError: If the specified axis is not 'x', 'y', or 'z'.

    Examples:
    --------
    # Create a cutting plane along the z-axis at z=2.5 with a half-size of 5
    cutting_plane_tag = create_cutting_plane('z', 2.5, 5)

    # Create a cutting plane along the x-axis at x=1.0 with a half-size of 3
    cutting_plane_tag = create_cutting_plane('x', 1.0, 3)
    """    
    if axis == 'x':
        planeTag = model.occ.addBox(level, -size, -size, 0.01, size * 2, size * 2)
    elif axis == 'y':
        planeTag = model.occ.addBox(-size, level, -size, size * 2, 0.01, size * 2)
    elif axis == 'z':
        planeTag = model.occ.addBox(-size, -size, level, size * 2, size * 2, 0.01)
    else:
        raise ValueError(f"{InternalLogger.pretty_function_details()}: Invalid axis: {axis}")
    
    check_tag(planeTag)
    planeDimTag = complete_dimtag('box', planeTag)
    check_dimtags(planeDimTag)
    
    if angle != 0.0:
        if axis == 'x':
            gmsh_rotate(planeDimTag, radians(angle), 0, 0)
        elif axis == 'y':
            gmsh_rotate(planeDimTag, 0, radians(angle), 0)
        else:
            gmsh_rotate(planeDimTag, 0, 0, radians(angle))
    
    model.occ.synchronize()
    return planeDimTag
