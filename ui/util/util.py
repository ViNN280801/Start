import tempfile

from vtk import vtkRenderer
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from styles import *
from constants import *
from numpy import all, array


def get_cur_datetime() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_thread_count():
    from multiprocessing import cpu_count

    return cpu_count()


def get_os_info():
    from platform import platform

    return platform()


def rad_to_degree(angle: float):
    from math import pi

    return angle * 180.0 / pi


def degree_to_rad(angle: float):
    from math import pi

    return angle * pi / 180.0


def is_mesh_dims(value: str):
    try:
        num = int(value)
        return num > 0 and num < 4
    except ValueError:
        return False


def is_path_ok(path: str):
    from os.path import exists, isfile

    if exists(path) and isfile(path):
        return True
    return False


def check_path_access(filename: str):
    from os import access, W_OK
    from os.path import dirname

    if not access(dirname(filename) or ".", W_OK):
        raise OSError(f"The path '{dirname(filename)}' is not accessible or writable.")


def ansi_to_segments(text: str):
    from re import split

    segments = []
    current_color = "light gray"  # Default color
    buffer = ""

    def append_segment(text, color):
        if text:  # Only append non-empty segments
            segments.append((text, color))

    # Split the text by ANSI escape codes
    parts = split(r"(\033\[\d+(?:;\d+)*m)", text)
    for part in parts:
        if not part:  # Skip empty strings
            continue
        if part.startswith("\033["):
            # Remove leading '\033[' and trailing 'm', then split
            codes = part[2:-1].split(";")
            for code in codes:
                if code in ANSI_TO_QCOLOR:
                    current_color = ANSI_TO_QCOLOR[code]
                    # Append the current buffer with the current color
                    append_segment(buffer, current_color)
                    buffer = ""  # Reset buffer
                    break  # Only apply the first matching color
        else:
            buffer += part  # Add text to the buffer
    append_segment(buffer, current_color)  # Append any remaining text
    return segments


def align_view_by_axis(
    axis: str, renderer: vtkRenderer, vtkWidget: QVTKRenderWindowInteractor
):
    axis = axis.strip().lower()

    if axis not in ["x", "y", "z", "center"]:
        return

    camera = renderer.GetActiveCamera()
    if axis == "x":
        camera.SetPosition(1, 0, 0)
        camera.SetViewUp(0, 0, 1)
    elif axis == "y":
        camera.SetPosition(0, 1, 0)
        camera.SetViewUp(0, 0, 1)
    elif axis == "z":
        camera.SetPosition(0, 0, 1)
        camera.SetViewUp(0, 1, 0)
    elif axis == "center":
        camera.SetPosition(1, 1, 1)
        camera.SetViewUp(0, 0, 1)

    camera.SetFocalPoint(0, 0, 0)

    renderer.ResetCamera()
    vtkWidget.GetRenderWindow().Render()


def calculate_direction(base, tip):
    from numpy import array
    from numpy.linalg import norm

    base = array(base, dtype=float)
    tip = array(tip, dtype=float)

    direction = tip - base

    norm_ = norm(direction)
    if norm_ == 0:
        raise ValueError("The direction vector cannot be zero.")
    direction /= norm_

    return direction


def calculate_thetaPhi(base, tip):
    from numpy import arctan2, arccos

    direction = calculate_direction(base, tip)
    x, y, z = direction[0], direction[1], direction[2]

    theta = arccos(z)
    phi = arctan2(y, x)

    return theta, phi


def calculate_thetaPhi_with_angles(x, y, z, angle_x, angle_y, angle_z):
    from numpy import array, cos, sin, arccos, arctan2, linalg, radians

    direction_vector = array(
        [
            cos(radians(angle_y)) * cos(radians(angle_z)),
            sin(radians(angle_x)) * sin(radians(angle_z)),
            cos(radians(angle_x)) * cos(radians(angle_y)),
        ]
    )
    norm = linalg.norm(direction_vector)
    theta = arccos(direction_vector[2] / norm)
    phi = arctan2(direction_vector[1], direction_vector[0])
    return theta, phi


def compute_distance_between_points(coord1, coord2):
    """
    Compute the Euclidean distance between two points in 3D space.
    """
    from math import sqrt
    from logger import InternalLogger

    try:
        result = sqrt(
            (coord1[0] - coord2[0]) ** 2
            + (coord1[1] - coord2[1]) ** 2
            + (coord1[2] - coord2[2]) ** 2
        )
    except Exception as e:
        print(InternalLogger.get_warning_none_result_with_exception_msg(e))
        return None
    return result


def can_create_plane(p1, p2):
    """
    Check if a plane can be created from two points.
    Returns True if the points can form a plane, otherwise raises an exception.

    Parameters:
    p1 (list or array): The first point in 3D space.
    p2 (list or array): The second point in 3D space.

    Returns:
    bool: True if a plane can be created, otherwise raises ValueError.

    Raises:
    ValueError: If the points are identical.
    """
    p1 = array(p1)
    p2 = array(p2)
    if all(p1 == p2):
        raise ValueError(f"Cannot create a plane with identical points: {p1}, {p2}")
    return True


def remove_last_occurrence(lst, item):
    """
    Remove the last occurrence of `item` from `lst`.
    If the item is not found, the function does nothing.

    Parameters:
    lst (list): The list from which to remove the item.
    item: The item to remove.
    """
    for i in range(len(lst) - 1, -1, -1):  # Iterate from the end to the beginning
        if lst[i] == item:
            del lst[i]  # Remove the item if found
            break  # Exit the loop after removing the item


def create_secure_tempfile() -> str:
    """
    Creates a secure temporary file and returns its file path.

    Returns:
    - str: Path to the securely created temporary file.
    """
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name

    return temp_file_path


def warning_unrealized_or_malfunctionating_function(functionality: str):
    """
    Prints to stdout warning message about breaked or unimplemented functionality.

    Parameters:
    functionality (str): String that presents functionality, for example: "Creating geometry object cone".

    Raises:
    An exception to prevent breaking program with unimplemented or bad, or dummy functionality.
    """
    if not functionality:
        raise ValueError("'functionality' param is empty, nothing to show")
    raise Exception(
        f"Warning: {functionality}: is unimplemented or breaken functionality."
    )
