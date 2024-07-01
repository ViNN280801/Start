from .geometry_constants import DEFAULT_CYLINDER_RESOLUTION


class Cylinder:
    """
    A class to represent a cylinder geometry.

    Attributes
    ----------
    x : float
        The x-coordinate of the cylinder's primary point.
    y : float
        The y-coordinate of the cylinder's primary point.
    z : float
        The z-coordinate of the cylinder's primary point.
    radius : float
        The radius of the cylinder.
    dx : float
        The length of the cylinder along the x-axis.
    dy : float
        The length of the cylinder along the y-axis.
    dz : float
        The length of the cylinder along the z-axis.
    resolution: int
        The resolution of the volume. (Its quality, with how many lines it will be created).
    mesh_resolution : int
        The triangle vtkLinearSubdivisionFilter count of the subdivisions.
    """

    def __init__(self,
                 x: float,
                 y: float,
                 z: float,
                 radius: float,
                 dx: float,
                 dy: float,
                 dz: float,
                 mesh_resolution: int,
                 resolution: int = DEFAULT_CYLINDER_RESOLUTION):
        """
        Constructs all the necessary attributes for the cylinder object.

        Parameters
        ----------
            x : float
                The x-coordinate of the cylinder's primary point.
            y : float
                The y-coordinate of the cylinder's primary point.
            z : float
                The z-coordinate of the cylinder's primary point.
            radius : float
                The radius of the cylinder.
            dx : float
                The length of the cylinder along the x-axis.
            dy : float
                The length of the cylinder along the y-axis.
            dz : float
                The length of the cylinder along the z-axis.
            mesh_resolution : int
                The triangle vtkLinearSubdivisionFilter count of the subdivisions.
        """
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.resolution = resolution
        self.mesh_resolution = mesh_resolution

    def __repr__(self):
        """
        Returns a string representation of the cylinder.

        Returns
        -------
        str
            A string representation of the cylinder.
        """
        cylinder_data_str = []
        cylinder_data_str.append(f'Primary Point: ({self.x}, {self.y}, {self.z})')
        cylinder_data_str.append(f'Radius: {self.radius}')
        cylinder_data_str.append(f'Length: {self.dx}')
        cylinder_data_str.append(f'Width: {self.dy}')
        cylinder_data_str.append(f'Height: {self.dz}')
        return '\n'.join(cylinder_data_str)
