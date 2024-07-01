from vtk import vtkPoints, vtkPolyLine, vtkCellArray, vtkPolyData
from sys import stderr


class Line:
    """
    A class to represent a line geometry.

    Attributes
    ----------
    points : list of tuple
        The list of points defining the line.
    """

    def __init__(self,
                 points: list):
        """
        Constructs all the necessary attributes for the line object.

        Parameters
        ----------
            log_console : LogConsole
                The logging console for outputting messages.
            points : list of tuple
                The list of points defining the line.
        """
        self.points = points

        if not Line.can_create_line(self.points):
            print(f"Can't create line with specified points:\n{self.points}", file=stderr)
            raise ValueError("Invalid points for creating a line.")            

    def __repr__(self):
        """
        Returns a string representation of the line.

        Returns
        -------
        str
            A string representation of the line.
        """
        points_str = [
            f'Point{i + 1}: ({x}, {y}, {z})'
            for i, (x, y, z) in enumerate(self.points)
        ]
        return '\n'.join(points_str)
    
    @staticmethod
    def can_create_line(point_data):
        """
        Check if a line can be created from the given set of points using VTK.

        Parameters:
        point_data (list of tuples): List of (x, y, z) coordinates of the points.

        Returns:
        bool: True if the line can be created, False otherwise.
        """
        # Check if all points are the same
        if all(point == point_data[0] for point in point_data):
            return False

        # Create a vtkPoints object and add the points
        points = vtkPoints()
        for x, y, z in point_data:
            points.InsertNextPoint(x, y, z)

        # Create a polyline object
        line = vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(point_data))

        for i in range(len(point_data)):
            line.GetPointIds().SetId(i, i)

        # Create a vtkCellArray and add the line to it
        lines = vtkCellArray()
        lines.InsertNextCell(line)

        # Create a polydata object and set the points and lines
        poly_data = vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetLines(lines)

        # Check if the line was created
        if poly_data.GetNumberOfLines() > 0:
            return True
        else:
            return False
