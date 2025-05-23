from vtk import (
    vtkRenderer,
    vtkPolyData,
    vtkPolyDataWriter,
    vtkAppendPolyData,
    vtkPolyDataReader,
    vtkPolyDataMapper,
    vtkActor,
    vtkPolyDataWriter,
)
from constants import (
    DEFAULT_TEMP_MESH_FILE,
    DEFAULT_TEMP_HDF5_FILE,
    DEFAULT_TEMP_VTK_FILE,
)
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from json import dump, load


class ProjectManager:
    @staticmethod
    def save_scene(
        renderer: vtkRenderer,
        logConsole,
        fontColor,
        colorbar_manager=None,
        actors_file="scene_actors.vtk",
        camera_file="scene_camera.json",
        colorbar_file="scene_colorbar.json",
    ):
        if (
            ProjectManager.save_actors(renderer, logConsole, fontColor, actors_file)
            is not None
            and ProjectManager.save_camera_settings(
                renderer, logConsole, fontColor, camera_file
            )
            is not None
        ):
            if colorbar_manager is not None:
                ProjectManager.save_colorbar_manager(
                    colorbar_manager, logConsole, colorbar_file
                )

            logConsole.insert_colored_text(
                "Successfully: ",
                f"Saved scene to the files: {actors_file}, {camera_file}, and {colorbar_file}",
                "green",
            )

    @staticmethod
    def save_actors(
        renderer: vtkRenderer, logConsole, fontColor, actors_file="scene_actors.vtk"
    ):
        try:
            append_filter = vtkAppendPolyData()
            actors_collection = renderer.GetActors()
            actors_collection.InitTraversal()

            for i in range(actors_collection.GetNumberOfItems()):
                actor = actors_collection.GetNextActor()
                if actor.GetMapper() and actor.GetMapper().GetInput():
                    poly_data = actor.GetMapper().GetInput()
                    if isinstance(poly_data, vtkPolyData):
                        append_filter.AddInputData(poly_data)

            append_filter.Update()

            writer = vtkPolyDataWriter()
            writer.SetFileName(actors_file)
            writer.SetInputData(append_filter.GetOutput())
            writer.Write()

            logConsole.insert_colored_text(
                "Info: ", f"Saved all actors to {actors_file}\n", "blue"
            )
            return 1
        except Exception as e:
            logConsole.insert_colored_text(
                "Error: ", f"Failed to save actors: {e}\n", "red"
            )
            return None

    @staticmethod
    def save_camera_settings(
        renderer: vtkRenderer, logConsole, fontColor, camera_file="scene_camera.json"
    ):
        from json import dump

        try:
            camera = renderer.GetActiveCamera()
            camera_settings = {
                "position": camera.GetPosition(),
                "focal_point": camera.GetFocalPoint(),
                "view_up": camera.GetViewUp(),
                "clip_range": camera.GetClippingRange(),
            }
            with open(camera_file, "w") as f:
                dump(camera_settings, f)

            return 1
        except Exception as e:
            logConsole.insert_colored_text(
                "Error: ", f"Failed to save camera settings: {e}\n", "red"
            )
            return None

    @staticmethod
    def load_scene(
        vtkWidget: QVTKRenderWindowInteractor,
        renderer: vtkRenderer,
        logConsole,
        fontColor,
        colorbar_manager=None,
        actors_file="scene_actors.vtk",
        camera_file="scene_camera.json",
        colorbar_file="scene_colorbar.json",
    ):
        if (
            ProjectManager.load_actors(renderer, logConsole, fontColor, actors_file)
            is not None
            and ProjectManager.load_camera_settings(
                renderer, logConsole, fontColor, camera_file
            )
            is not None
        ):
            if colorbar_manager is not None:
                loaded_colorbar_manager = ProjectManager.load_colorbar_manager(
                    vtkWidget, renderer, logConsole, colorbar_file
                )
                if loaded_colorbar_manager:
                    colorbar_manager = loaded_colorbar_manager

            vtkWidget.GetRenderWindow().Render()
            logConsole.insert_colored_text(
                "Successfully: ",
                f"Loaded scene from the files: {actors_file}, {camera_file}, and {colorbar_file}",
                "green",
            )

    @staticmethod
    def load_actors(
        renderer: vtkRenderer, logConsole, fontColor, actors_file="scene_actors.vtk"
    ):
        try:
            reader = vtkPolyDataReader()
            reader.SetFileName(actors_file)
            reader.Update()

            mapper = vtkPolyDataMapper()
            mapper.SetInputData(reader.GetOutput())

            actor = vtkActor()
            actor.SetMapper(mapper)
            # There we don't need to add actor, because it added from the config file (config file contains mesh file)
            renderer.ResetCamera()

            logConsole.insert_colored_text(
                "Info: ", f"Loaded actors from {actors_file}\n", "blue"
            )
            return 1
        except Exception as e:
            logConsole.insert_colored_text(
                "Error: ", f"Failed to load actors: {e}\n", "red"
            )
            return None

    @staticmethod
    def load_camera_settings(
        renderer: vtkRenderer, logConsole, fontColor, camera_file="scene_camera.json"
    ):
        from json import load

        try:
            with open(camera_file, "r") as f:
                camera_settings = load(f)

            camera = renderer.GetActiveCamera()
            camera.SetPosition(*camera_settings["position"])
            camera.SetFocalPoint(*camera_settings["focal_point"])
            camera.SetViewUp(*camera_settings["view_up"])
            camera.SetClippingRange(*camera_settings["clip_range"])

            renderer.ResetCamera()
            return 1
        except Exception as e:
            logConsole.insert_colored_text(
                "Error: ", f"Failed to load camera settings: {e}\n", "red"
            )
            return None

    @staticmethod
    def remove_temp_files_helper(filename: str):
        from os import remove
        from os.path import exists

        try:
            if exists(filename):
                remove(filename)
        except Exception as ex:
            print(f"Some error occurs: Can't remove file {filename}. Error: {ex}")
            return

    @staticmethod
    def remove_temp_files():
        # High probability that user don't want to delete temporary config file

        # Removing all temporary files excluding temporary config file
        ProjectManager.remove_temp_files_helper(DEFAULT_TEMP_MESH_FILE)
        ProjectManager.remove_temp_files_helper(DEFAULT_TEMP_VTK_FILE)
        ProjectManager.remove_temp_files_helper(DEFAULT_TEMP_HDF5_FILE)

    @staticmethod
    def save_colorbar_manager(
        colorbar_manager, logConsole, colorbar_file="scene_colorbar.json"
    ):
        try:
            properties = colorbar_manager.get_properties()
            with open(colorbar_file, "w") as f:
                dump(properties, f)
            logConsole.insert_colored_text(
                "Info: ", f"Saved colorbar manager to {colorbar_file}", "blue"
            )
            return 1
        except Exception as e:
            logConsole.insert_colored_text(
                "Error: ", f"Failed to save colorbar manager: {e}", "red"
            )
            return None

    @staticmethod
    def load_colorbar_manager(
        vtkWidget, renderer, logConsole, colorbar_file="scene_colorbar.json"
    ):
        from tabs import ColorbarManager

        try:
            with open(colorbar_file, "r") as f:
                properties = load(f)
            colorbar_manager = ColorbarManager.from_properties(
                vtkWidget, renderer, properties
            )
            logConsole.insert_colored_text(
                "Info: ", f"Loaded colorbar manager from {colorbar_file}", "blue"
            )
            return colorbar_manager
        except Exception as e:
            logConsole.insert_colored_text(
                "Error: ", f"Failed to load colorbar manager: {e}", "red"
            )
            return None
