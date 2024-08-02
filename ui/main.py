import os
import sys

from util.inst_deps import check_and_install_packages
check_and_install_packages(["numpy", "h5py", "gmsh", "matplotlib", "PyQt5", "vtk", "nlohmann-json", "psutil"])

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "constants")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "data")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "dialogs")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "field_validators")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "logger")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "styles")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "tabs")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "tests")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "util")))

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
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Application crushed by reason: {e}", file=sys.stderr)
    finally:
        gmsh_finalize()


if __name__ == "__main__":
    main()
