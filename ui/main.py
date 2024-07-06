from util.inst_deps import check_and_install_packages
check_and_install_packages(["numpy", "h5py", "gmsh", "matplotlib", "PyQt5", "vtk", "nlohmann-json", "psutil"])


import sys
from PyQt5.QtWidgets import QApplication
from window import WindowApp
from util.gmsh_helpers import gmsh_init, gmsh_finalize


def main():
    gmsh_init() # Initializing gmsh session only once for all runtime of the appliction
    
    app = QApplication(sys.argv)
    main_window = WindowApp()
    main_window.show()
    
    # Handling app crush
    try:
        exit(app.exec_())
    except Exception as e:
        print(f"Application crushed by reason: {e}", file=sys.stderr)
    finally:
        gmsh_finalize()


if __name__ == "__main__":
    main()
