import ast
import os
import argparse


def get_imports(filepath):
    """
    Parse the given Python file to extract all import statements.

    Args:
        filepath (str): The path to the Python file.

    Returns:
        set: A set of module names imported in the file.
    """
    imports = set()
    with open(filepath, "r", encoding="utf-8") as file:
        tree = ast.parse(file.read(), filename=filepath)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
    return imports


def find_py_files(start_path):
    """
    Recursively find all Python files in the given directory.

    Args:
        start_path (str): The starting directory to search for Python files.

    Returns:
        list: A list of paths to Python files found.
    """
    py_files = []
    for root, _, files in os.walk(start_path):
        for file in files:
            if file.endswith(".py"):
                py_files.append(os.path.join(root, file))
    return py_files


def find_all_imports(start_path):
    """
    Find all unique imports in all Python files within the given directory.

    Args:
        start_path (str): The starting directory to search for Python files and imports.

    Returns:
        set: A set of all unique module names imported in the directory.
    """
    all_imports = set()
    py_files = find_py_files(start_path)
    seen_files = set()
    for filepath in py_files:
        if filepath not in seen_files:
            seen_files.add(filepath)
            imports = get_imports(filepath)
            all_imports.update(imports)
    return all_imports


def is_local_module(module, start_path):
    """
    Check if the given module is a local module within the project.

    Args:
        module (str): The name of the module.
        start_path (str): The starting directory of the project.

    Returns:
        bool: True if the module is local, False otherwise.
    """
    module_parts = module.split('.')
    for i in range(1, len(module_parts) + 1):
        module_path = os.path.join(start_path, *module_parts[:i])
        if os.path.isfile(module_path + ".py") or os.path.isfile(os.path.join(module_path, "__init__.py")):
            return True
    return False


def find_local_modules(start_path):
    """
    Find all local modules by looking for __init__.py files within the project.

    Args:
        start_path (str): The starting directory of the project.

    Returns:
        set: A set of local module names found in the project.
    """
    local_modules = set()
    py_files = find_py_files(start_path)
    for filepath in py_files:
        if filepath.endswith("__init__.py"):
            imports = get_imports(filepath)
            for imp in imports:
                if imp.startswith('.'):
                    imp = imp[1:]
                local_modules.add(imp)
    return local_modules


def find_submodules(module, start_path):
    """
    Find all submodules within the given module's directory.

    Args:
        module (str): The name of the module.
        start_path (str): The starting directory of the project.

    Returns:
        set: A set of submodule names found within the module's directory.
    """
    submodules = set()
    module_path = os.path.join(start_path, *module.split('.'))
    if os.path.isdir(module_path):
        for root, _, files in os.walk(module_path):
            for file in files:
                if file.endswith(".py") and file != "__init__.py":
                    submodule = os.path.relpath(os.path.join(root, file), start_path).replace(os.path.sep, '.').rsplit('.', 1)[0]
                    submodules.add(submodule)
    return submodules


def find_all_local_submodules(local_modules, start_path):
    """
    Find all submodules for all local modules in the project.

    Args:
        local_modules (set): A set of local module names.
        start_path (str): The starting directory of the project.

    Returns:
        set: A set of all submodule names found within the local modules.
    """
    all_submodules = set()
    for module in local_modules:
        submodules = find_submodules(module, start_path)
        all_submodules.update(submodules)
    return all_submodules


def print_imports(all_imports, local_modules, local_submodules, start_path, show_embedded_imports=False):
    """
    Print the list of all imports in the project, with local modules colorized in blue.
    Optionally, show embedded imports.

    Args:
        all_imports (set): A set of all unique imports in the project.
        local_modules (set): A set of local module names.
        local_submodules (set): A set of local submodule names.
        start_path (str): The starting directory of the project.
        show_embedded_imports (bool): Whether to show embedded imports. Defaults to False.
    """
    own_modules_count = 0
    imported_modules_count = 0
    sorted_imports = sorted(all_imports)
    printed_imports = set()
    print("All imports in the project:")
    for imp in sorted_imports:
        parts = imp.split('.')
        colorized_parts = []
        for i, part in enumerate(parts):
            full_module = '.'.join(parts[:i+1])
            if full_module in local_modules or full_module in local_submodules or is_local_module(full_module, start_path):
                colorized_parts.append(f"\033[34m{part}\033[0m")
                if i == 0:  # Only count top-level parts as own modules
                    own_modules_count += 1
            else:
                colorized_parts.append(part)
                if i == 0:  # Only count top-level parts as imported modules
                    imported_modules_count += 1
        # Print only top-level part if it's a local module, otherwise print full import
        if parts[0] not in printed_imports:
            if is_local_module(parts[0], start_path):
                print(f"\033[34m{parts[0]}\033[0m")
            else:
                print('.'.join(colorized_parts))
            printed_imports.add(parts[0])
    print(f"\nOwn modules: {own_modules_count}")
    print(f"Imported modules: {imported_modules_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find all imports in a Python project.")
    parser.add_argument("start_path", help="The starting directory for import analysis.")
    args = parser.parse_args()
    start_path = args.start_path

    all_imports = find_all_imports(start_path)
    local_modules = find_local_modules(start_path)
    local_submodules = find_all_local_submodules(local_modules, start_path)

    all_imports.discard(None)

    print_imports(all_imports, local_modules, local_submodules, start_path)
