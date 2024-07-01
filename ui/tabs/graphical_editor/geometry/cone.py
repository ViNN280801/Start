class Cone:
    """
    A class to represent a cone geometry.

    Attributes
    ----------
    x : float
        The x-coordinate of the cone's base center.
    y : float
        The y-coordinate of the cone's base center.
    z : float
        The z-coordinate of the cone's base center.
    dx : float
        The x-component of the cone's axis direction.
    dy : float
        The y-component of the cone's axis direction.
    dz : float
        The z-component of the cone's axis direction.
    r : float
        The radius of the cone's base.
    mesh_resolution : int
        The resolution of the cone's mesh.
    """

    def __init__(self, 
                 x: float, 
                 y: float, 
                 z: float, 
                 dx: float, 
                 dy: float, 
                 dz: float, 
                 r: float, 
                 resolution: int, 
                 mesh_resolution: int):
        """
        Constructs all the necessary attributes for the cone object.

        Parameters
        ----------
            x : float
                The x-coordinate of the cone's base center.
            y : float
                The y-coordinate of the cone's base center.
            z : float
                The z-coordinate of the cone's base center.
            dx : float
                The x-component of the cone's axis direction.
            dy : float
                The y-component of the cone's axis direction.
            dz : float
                The z-component of the cone's axis direction.
            r : float
                The radius of the cone's base.
            resolution: int
                The resolution of the volume. (Its quality, with how many lines it will be created).
            mesh_resolution : int
                The triangle vtkLinearSubdivisionFilter count of the subdivisions.
        """
        self.x = x
        self.y = y
        self.z = z
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.height = (self.dx**2 + self.dy**2 + self.dz**2)**0.5
        self.r = r
        self.resolution = resolution
        self.mesh_resolution = mesh_resolution

    def __repr__(self):
        """
        Returns a string representation of the cone.

        Returns
        -------
        str
            A string representation of the cone.
        """
        cone_data_str = []
        cone_data_str.append(f'Base Center: ({self.x}, {self.y}, {self.z})')
        cone_data_str.append(f'Axis Direction: ({self.dx}, {self.dy}, {self.dz})')
        cone_data_str.append(f'Height: {self.height}')
        cone_data_str.append(f'Base Radius: {self.r}')
        return '\n'.join(cone_data_str)
