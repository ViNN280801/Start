from vtk import vtkPoints, vtkPolyData, vtkDelaunay2D
from sys import stderr


class Surface:
    """
    A class to represent a surface geometry.

    Attributes
    ----------
    log_console : LogConsole
        The logging console for outputting messages.
    points : list of tuple
        The list of points defining the surface.
    """

    def __init__(self, points: list):
        """
        Constructs all the necessary attributes for the surface object.

        Parameters
        ----------
            points : list of tuple
                The list of points defining the surface.
        """
        self.points = points

    def __repr__(self):
        """
        Returns a string representation of the surface.

        Returns
        -------
        str
            A string representation of the surface.
        """
        points_str = [
            f"Point{i + 1}: ({x}, {y}, {z})" for i, (x, y, z) in enumerate(self.points)
        ]
        return "\n".join(points_str)
