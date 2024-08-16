from os.path import exists, isfile


def is_file_valid(path: str) -> bool:
    """
    Check if the provided path points to a valid file.

    :param path: The path to check.
    :return: True if the path exists, is a file, and is not an empty string; False otherwise.
    """
    if not exists(path) or not isfile(path) or not path:
        return False
    return True


def is_path_accessible(path: str) -> bool:
    """
    Check if the provided path is accessible (can be opened).

    :param path: The path to check.
    :return: True if the path is accessible; False otherwise.
    """
    if not path:
        return False
    
    try:
        with open(path) as _:
            pass
        return True
    except IOError:
        return False


def check_path(path: str):
    """
    Validate and ensure that the provided path is both valid and accessible.

    This function uses the `is_file_valid()` and `is_path_accessible()` functions.
    If the path is invalid or inaccessible, it raises an appropriate exception.

    :param path: The path to check.
    :raises ValueError: If the path is not valid.
    :raises IOError: If the path is not accessible.
    """
    if not is_file_valid(path):
        raise ValueError(f"The path '{path}' is not valid.")
    
    if not is_path_accessible(path):
        raise IOError(f"The path '{path}' is not accessible.")
