import re
from util.path_file_checkers import check_path


class PosFileParser:
    def __init__(self, filename):
        """
        Initialize the PosFileParser with the given filename.

        :param filename: The name of the file to parse.
        """
        check_path(filename)
        
        self.filename = filename
        self.points = []
        self.vectors = []

    def parse(self):
        """
        Parses the file to extract points and vectors.

        The file is expected to have lines with the format:
        VP(x,y,z){vx,vy,vz};

        Where x, y, z are the coordinates of a point, and vx, vy, vz
        are the components of a vector.

        :return: A tuple containing two lists:
                 - A list of points, where each point is a tuple (x, y, z).
                 - A list of vectors, where each vector is a tuple (vx, vy, vz).
        :raises ValueError: If the file contains no valid VP data.
        :raises FileNotFoundError: If the file is not found.
        :raises Exception: If an unexpected error occurs while parsing the file.
        """
        try:
            with open(self.filename, 'r') as file:
                lines = file.readlines()
                
                for line in lines:
                    # Match the pattern for the point and vector data
                    match = re.match(r"VP\(([^)]+)\)\{([^\}]+)\};", line)
                    if match:
                        # Extract the point data
                        point_str = match.group(1)
                        x, y, z = map(float, point_str.split(','))
                        self.points.append((x, y, z))

                        # Extract the vector data
                        vector_str = match.group(2)
                        vx, vy, vz = map(float, vector_str.split(','))
                        self.vectors.append((vx, vy, vz))

            # Check if any points or vectors were found
            if not self.points or not self.vectors:
                raise ValueError(f"The file '{self.filename}' contains no valid VP data.")

            return self.points, self.vectors

        except FileNotFoundError as e:
            raise FileNotFoundError(f"File Error: File '{self.filename}' not found.")

        except Exception as e:
            raise Exception(f"Parsing Error: An error occurred while parsing the file: {e}")
