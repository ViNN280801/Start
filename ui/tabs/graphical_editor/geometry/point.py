class Point:
    """
    A class to represent a point geometry.

    Attributes
    ----------
    x : float
        The x-coordinate of the point.
    y : float
        The y-coordinate of the point.
    z : float
        The z-coordinate of the point.
    """

    def __init__(self, x: float, y: float, z: float):
        """
        Constructs all the necessary attributes for the point object.

        Parameters
        ----------
            log_console : LogConsole
                The logging console for outputting messages.
            x : float
                The x-coordinate of the point.
            y : float
                The y-coordinate of the point.
            z : float
                The z-coordinate of the point.
        """
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        """
        Returns a string representation of the point.

        Returns
        -------
        str
            A string representation of the point.
        """
        return f"Point: ({self.x}, {self.y}, {self.z})"
