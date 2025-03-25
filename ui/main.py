import sys
from PyQt5.QtWidgets import QApplication
from window import WindowApp
from util.gmsh_helpers import gmsh_init, gmsh_finalize


def main():
    gmsh_init()  # Initializing gmsh session only once for all runtime of the appliction

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
