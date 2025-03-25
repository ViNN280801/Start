class Box:
    """
    A class to represent a box geometry.

    Attributes
    ----------
    x : float
        The x-coordinate of the box's primary point.
    y : float
        The y-coordinate of the box's primary point.
    z : float
        The z-coordinate of the box's primary point.
    length : float
        The length of the box.
    width : float
        The width of the box.
    height : float
        The height of the box.
    mesh_resolution : int
        The triangle vtkLinearSubdivisionFilter count of the subdivisions.
    """

    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        length: float,
        width: float,
        height: float,
        mesh_resolution: int,
    ):
        """
        Constructs all the necessary attributes for the box object.

        Parameters
        ----------
            x : float
                The x-coordinate of the box's primary point.
            y : float
                The y-coordinate of the box's primary point.
            z : float
                The z-coordinate of the box's primary point.
            length : float
                The length of the box.
            width : float
                The width of the box.
            height : float
                The height of the box.
            mesh_resolution : int
                The triangle vtkLinearSubdivisionFilter count of the subdivisions.
        """
        self.x = x
        self.y = y
        self.z = z
        self.length = length
        self.width = width
        self.height = height
        self.mesh_resolution = mesh_resolution

    def __repr__(self):
        """
        Returns a string representation of the box.

        Returns
        -------
        str
            A string representation of the box.
        """
        box_data_str = []
        box_data_str.append(f"Primary Point: ({self.x}, {self.y}, {self.z})")
        box_data_str.append(f"Length: {self.length}")
        box_data_str.append(f"Width: {self.width}")
        box_data_str.append(f"Height: {self.height}")
        return "\n".join(box_data_str)
