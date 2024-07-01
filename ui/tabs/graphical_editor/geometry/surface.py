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

    def __init__(self,
                 points: list):
        """
        Constructs all the necessary attributes for the surface object.

        Parameters
        ----------
            points : list of tuple
                The list of points defining the surface.
        """
        self.points = points

        if not Surface.can_create_surface(self.points):
            print(f"Can't create surface with specified points:\n{self.points}", file=stderr)
            raise ValueError("Invalid points for creating a surface.")

    def __repr__(self):
        """
        Returns a string representation of the surface.

        Returns
        -------
        str
            A string representation of the surface.
        """
        points_str = [
            f'Point{i + 1}: ({x}, {y}, {z})'
            for i, (x, y, z) in enumerate(self.points)
        ]
        return '\n'.join(points_str)
    
    @staticmethod
    def can_create_surface(point_data):
        """
        Check if a surface can be created from the given set of points using VTK.

        Parameters:
        point_data (list of tuples): List of (x, y, z) coordinates of the points.

        Returns:
        bool: True if the surface can be created, False otherwise.
        """
        # Create a vtkPoints object and add the points
        points = vtkPoints()
        for x, y, z in point_data:
            points.InsertNextPoint(x, y, z)

        # Create a polydata object and set the points
        poly_data = vtkPolyData()
        poly_data.SetPoints(points)

        # Create a Delaunay2D object and set the input
        delaunay = vtkDelaunay2D()
        delaunay.SetInputData(poly_data)

        # Try to create the surface
        delaunay.Update()

        # Check if the surface was created
        output = delaunay.GetOutput()
        if output.GetNumberOfCells() > 0:
            return True
        else:
            return False
