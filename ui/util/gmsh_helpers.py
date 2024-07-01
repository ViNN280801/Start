from gmsh import initialize, finalize, isInitialized, model


def gmsh_init():
    """Initialize Gmsh if it is not already initialized."""
    if not isInitialized():
        initialize()


def gmsh_finalize():
    """Finalize Gmsh if it is initialized."""
    if isInitialized():
        finalize()


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
    for dim, _ in dimtags:
        if dim != 3:
            raise ValueError(f"Invalid dimension {dim} in dimtags. Only dimension 3 is supported.")


def complete_dimtag(geometry: str, tag: int):
    """
    Compose dimtag from one tag and geometry string representation.
    
    Args:
    geometry (str): The string representation of the geometry.
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


def create_cutting_plane(axis: str, level: float, size: float):
    """
    Creates a cutting plane along a specified axis at a given level.

    Parameters:
    axis (str): The axis along which to create the cutting plane ('x', 'y', or 'z').
    level (float): The position along the specified axis where the cutting plane will be created.
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
        return model.occ.addBox(level, -size, -size, 0.01, size * 2, size * 2)
    elif axis == 'y':
        return model.occ.addBox(-size, level, -size, size * 2, 0.01, size * 2)
    elif axis == 'z':
        return model.occ.addBox(-size, -size, level, size * 2, size * 2, 0.01)
    else:
        raise ValueError("Invalid axis")
