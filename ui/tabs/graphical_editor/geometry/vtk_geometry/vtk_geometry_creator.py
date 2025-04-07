from vtk import (
    vtkPoints,
    vtkVertexGlyphFilter,
    vtkPolyLine,
    vtkCellArray,
    vtkPolygon,
    vtkSphereSource,
    vtkCubeSource,
    vtkConeSource,
    vtkCylinderSource,
    vtkActor,
    vtkPolyData,
    vtkPolyDataMapper,
    vtkTriangleFilter,
    vtkLinearSubdivisionFilter,
    vtkTransform,
    vtkTransformFilter,
    vtkPlane,
    vtkPlaneSource,
)
from sys import stderr
from tabs.graphical_editor.geometry.point import Point
from tabs.graphical_editor.geometry.line import Line
from tabs.graphical_editor.geometry.surface import Surface
from tabs.graphical_editor.geometry.sphere import Sphere
from tabs.graphical_editor.geometry.box import Box
from tabs.graphical_editor.geometry.cone import Cone
from tabs.graphical_editor.geometry.cylinder import Cylinder

from numpy import array, cross, dot, arccos, pi
from numpy.linalg import norm
from util.vtk_helpers import remove_actor
from util.util import can_create_plane


class VTKGeometryCreator:
    @staticmethod
    def remove(vtkWidget, renderer, actor: vtkActor, needResetCamera: bool = True):
        """
        Remove an actor from a VTK renderer and optionally reset the camera.

        Parameters:
        vtkWidget : vtkRenderWindowInteractor
            The VTK widget that contains the render window.
        renderer : vtkRenderer
            The renderer from which the actor will be removed.
        actor : vtkActor
            The actor to be removed from the renderer.
        needResetCamera : bool, optional
            If True, the camera will be reset after removing the actor (default is True).

        """
        remove_actor(vtkWidget, renderer, actor, needResetCamera)

    @staticmethod
    def create_point(point: Point) -> vtkActor:
        """
        Creates the point using VTK and returns the actor.

        Returns
        -------
        vtkActor
            The actor representing the point.
        """
        try:
            vtk_points = vtkPoints()
            vtk_points.InsertNextPoint(point.x, point.y, point.z)

            poly_data = vtkPolyData()
            poly_data.SetPoints(vtk_points)

            glyph_filter = vtkVertexGlyphFilter()
            glyph_filter.SetInputData(poly_data)
            glyph_filter.Update()

            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(glyph_filter.GetOutputPort())

            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetPointSize(10)

            return actor

        except Exception as e:
            print(
                f"An error occurred while creating the point with VTK: {e}", file=stderr
            )

    @staticmethod
    def create_line(line: Line) -> vtkActor:
        """
        Creates the line using VTK and returns the actor.

        Returns
        -------
        vtkActor
            The actor representing the line.
        """
        try:
            vtk_points = vtkPoints()
            polyline = vtkPolyLine()
            polyline.GetPointIds().SetNumberOfIds(len(line.points))

            for i, (x, y, z) in enumerate(line.points):
                vtk_points.InsertNextPoint(x, y, z)
                polyline.GetPointIds().SetId(i, i)

            cells = vtkCellArray()
            cells.InsertNextCell(polyline)

            poly_data = vtkPolyData()
            poly_data.SetPoints(vtk_points)
            poly_data.SetLines(cells)

            mapper = vtkPolyDataMapper()
            mapper.SetInputData(poly_data)

            actor = vtkActor()
            actor.SetMapper(mapper)

            return actor

        except Exception as e:
            print(
                f"An error occurred while creating the line with VTK: {e}", file=stderr
            )

    @staticmethod
    def create_surface(surface: Surface) -> vtkActor:
        """
        Creates the surface using VTK and returns the actor.

        Returns
        -------
        vtkActor
            The actor representing the surface.
        """
        try:
            vtk_points = vtkPoints()
            polygon = vtkPolygon()
            polygon.GetPointIds().SetNumberOfIds(len(surface.points))

            for i, (x, y, z) in enumerate(surface.points):
                vtk_points.InsertNextPoint(x, y, z)
                polygon.GetPointIds().SetId(i, i)

            cells = vtkCellArray()
            cells.InsertNextCell(polygon)

            poly_data = vtkPolyData()
            poly_data.SetPoints(vtk_points)
            poly_data.SetPolys(cells)

            mapper = vtkPolyDataMapper()
            mapper.SetInputData(poly_data)

            actor = vtkActor()
            actor.SetMapper(mapper)

            return actor

        except Exception as e:
            print(
                f"An error occurred while creating the surface with VTK: {e}",
                file=stderr,
            )

    @staticmethod
    def create_plane(p1, p2, axis="z"):
        """
        Create a cutting plane in VTK defined by two points and an axis.

        Args:
            p1 (list): Coordinates of the first point [x, y, z].
            p2 (list): Coordinates of the second point [x, y, z].
            axis (str): The axis along which to extrude the plane ('x', 'y', 'z'). Default is 'z'.

        Returns:
            vtkPlane: The created VTK plane. (`vtkPlane` type).

        Raises:
            ValueError: If the inputs are not valid.
            RuntimeError: If there is an error during the plane creation.
        """
        if not (
            isinstance(p1, (list, set, tuple))
            and isinstance(p2, (list, set, tuple))
            and len(p1) == 3
            and len(p2) == 3
        ):
            raise ValueError(
                "Both points must be lists/sets/tuples of three numerical coordinates."
            )
        if not all(isinstance(coord, (int, float)) for coord in p1 + p2):
            raise ValueError("All coordinates must be integers or floats.")
        if axis not in ["x", "y", "z"]:
            raise ValueError("Selected axis must be 'x', 'Y', or 'z'.")

        can_create_plane(p1, p2)

        try:
            direction = [
                p2[i] - p1[i] for i in range(3)
            ]  # Direction vector of the line
            plane = vtkPlane()
            viewDirection = [0, 0, 0]

            # Set the view direction based on the selected axis
            if axis == "x":
                viewDirection = [1, 0, 0]
            elif axis == "Y":
                viewDirection = [0, 1, 0]
            elif axis == "z":
                viewDirection = [0, 0, 1]

            normal = cross(direction, viewDirection)
            plane.SetOrigin(p1)
            plane.SetNormal(normal)

            plane_source = vtkPlaneSource()
            plane_source.SetOrigin(p1)
            plane_source.SetPoint1(
                p1[0] + viewDirection[0],
                p1[1] + viewDirection[1],
                p1[2] + viewDirection[2],
            )
            plane_source.SetPoint2(
                p2[0] + viewDirection[0],
                p2[1] + viewDirection[1],
                p2[2] + viewDirection[2],
            )
            plane_source.Update()

            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(plane_source.GetOutputPort())

            actor = vtkActor()
            actor.SetMapper(mapper)

            return plane, actor

        except Exception as e:
            raise RuntimeError(f"Error creating plane in VTK: {e}")

    @staticmethod
    def create_sphere(sphere: Sphere) -> vtkActor:
        """
        Creates the sphere using VTK and returns the actor.

        Returns
        -------
        vtkActor
            The actor representing the sphere.
        """
        try:
            sphere_source = vtkSphereSource()
            sphere_source.SetCenter(sphere.x, sphere.y, sphere.z)
            sphere_source.SetRadius(sphere.radius)
            sphere_source.Update()
            sphere_source.SetPhiResolution(sphere.phi_resolution)
            sphere_source.SetThetaResolution(sphere.theta_resolution)

            triangle_filter = vtkTriangleFilter()
            triangle_filter.SetInputConnection(sphere_source.GetOutputPort())
            triangle_filter.Update()

            subdivision_filter = vtkLinearSubdivisionFilter()
            subdivision_filter.SetInputConnection(triangle_filter.GetOutputPort())
            subdivision_filter.SetNumberOfSubdivisions(sphere.mesh_resolution)
            subdivision_filter.Update()

            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(subdivision_filter.GetOutputPort())

            actor = vtkActor()
            actor.SetMapper(mapper)

            return actor

        except Exception as e:
            print(
                f"An error occurred while creating the sphere with VTK: {e}",
                file=stderr,
            )

    @staticmethod
    def create_box(box: Box) -> vtkActor:
        """
        Creates the box using VTK and returns the actor.

        Returns
        -------
        vtkActor
            The actor representing the box.
        """
        try:
            cube_source = vtkCubeSource()

            x_min = min(box.x, box.x + box.length)
            x_max = max(box.x, box.x + box.length)
            y_min = min(box.y, box.y + box.width)
            y_max = max(box.y, box.y + box.width)
            z_min = min(box.z, box.z + box.height)
            z_max = max(box.z, box.z + box.height)

            cube_source.SetBounds(x_min, x_max, y_min, y_max, z_min, z_max)
            cube_source.Update()

            triangle_filter = vtkTriangleFilter()
            triangle_filter.SetInputConnection(cube_source.GetOutputPort())
            triangle_filter.Update()

            subdivision_filter = vtkLinearSubdivisionFilter()
            subdivision_filter.SetInputConnection(triangle_filter.GetOutputPort())
            subdivision_filter.SetNumberOfSubdivisions(box.mesh_resolution)
            subdivision_filter.Update()

            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(subdivision_filter.GetOutputPort())

            actor = vtkActor()
            actor.SetMapper(mapper)

            return actor

        except Exception as e:
            print(
                f"An error occurred while creating the box with VTK: {e}", file=stderr
            )
            return None

    @staticmethod
    def create_cone(cone: Cone) -> vtkActor:
        """
        Creates the cone using VTK and returns the actor.

        Returns
        -------
        vtkActor
            The actor representing the cone.
        """
        try:
            cone_source = vtkConeSource()
            cone_source.SetCenter(cone.x_center, cone.y_center, cone.z_center)
            cone_source.SetDirection(cone.dx, cone.dy, cone.dz)
            cone_source.SetRadius(cone.r)
            cone_source.SetHeight(cone.height)
            cone_source.SetResolution(cone.resolution)
            cone_source.Update()

            triangle_filter = vtkTriangleFilter()
            triangle_filter.SetInputConnection(cone_source.GetOutputPort())
            triangle_filter.Update()

            subdivision_filter = vtkLinearSubdivisionFilter()
            subdivision_filter.SetInputConnection(triangle_filter.GetOutputPort())
            subdivision_filter.SetNumberOfSubdivisions(cone.mesh_resolution)
            subdivision_filter.Update()

            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(subdivision_filter.GetOutputPort())

            actor = vtkActor()
            actor.SetMapper(mapper)

            return actor

        except Exception as e:
            print(
                f"An error occurred while creating the cone with VTK: {e}", file=stderr
            )
            return None

    @staticmethod
    def create_cylinder(cylinder: Cylinder) -> vtkActor:
        """
        Creates the cylinder using VTK and returns the actor.

        Returns
        -------
        vtkActor
            The actor representing the cylinder.
        """
        try:
            cylinder_source = vtkCylinderSource()
            cylinder_source.SetCenter(cylinder.x, cylinder.y, cylinder.z)
            cylinder_source.SetRadius(cylinder.radius)
            cylinder_source.SetHeight(cylinder.height)
            cylinder_source.SetResolution(cylinder.resolution)
            cylinder_source.Update()

            triangle_filter = vtkTriangleFilter()
            triangle_filter.SetInputConnection(cylinder_source.GetOutputPort())
            triangle_filter.Update()

            subdivision_filter = vtkLinearSubdivisionFilter()
            subdivision_filter.SetInputConnection(triangle_filter.GetOutputPort())
            subdivision_filter.SetNumberOfSubdivisions(cylinder.mesh_resolution)
            subdivision_filter.Update()

            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(subdivision_filter.GetOutputPort())

            actor = vtkActor()
            actor.SetMapper(mapper)

            sync_vtkcylinder_to_gmshcylinder_helper(cylinder, actor)

            return actor

        except Exception as e:
            print(
                f"An error occurred while creating the cylinder with VTK: {e}",
                file=stderr,
            )


def sync_vtkcylinder_to_gmshcylinder_helper(cylinder: Cylinder, actor: vtkActor):
    """
    Adjusts a VTK cylinder actor to match the orientation and position of a Gmsh-defined cylinder.

    Parameters:
    -----------
    cylinder : Cylinder
        An object containing the cylinder parameters (x, y, z, radius, height, resolution, dx, dy, dz).
    actor : vtkActor
        The VTK actor representing the cylinder to be adjusted.

    Notes:
    ------
    This function applies both translation and rotation transformations to align the VTK cylinder with
    the Gmsh-defined cylinder's direction vector and position. It also updates the geometry of the actor
    to reflect these transformations.

    The transformation steps include:
    1. Translating the cylinder to its base position.
    2. Calculating the necessary rotation to align the default Y-axis with the direction vector.
    3. Applying the rotation and additional translation to position the cylinder correctly.
    4. Updating the actor's transformation matrix and geometry.
    """
    # Calculate the direction and height
    direction = array([cylinder.dx, cylinder.dy, cylinder.dz])
    direction_normalized = direction / cylinder.height

    # Translate to the cylinder base
    transform = vtkTransform()
    transform.Translate(cylinder.x, cylinder.y, cylinder.z)

    # Calculate the rotation axis and angle
    # Relevant links:
    #        1) https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    #        2) https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    #        3) https://en.wikipedia.org/wiki/Euler%27s_rotation_theorem
    default_direction = array([0, 1, 0])  # Default Y-axis in VTK
    rotation_axis = cross(default_direction, direction_normalized)
    rotation_angle = arccos(dot(default_direction, direction_normalized)) * 180 / pi

    # Apply rotation if the rotation axis is non-zero
    if norm(rotation_axis) > 1e-6:
        transform.RotateWXYZ(
            rotation_angle, rotation_axis[0], rotation_axis[1], rotation_axis[2]
        )

    # Translate along the direction vector to position the cylinder correctly
    transform.Translate(0, cylinder.height / 2, 0)

    # Applying transformation not only for te actor, but for its geometry too
    if actor and isinstance(actor, vtkActor):
        mapper = actor.GetMapper()
        if mapper:
            input_data = mapper.GetInput()
            transform_filter = vtkTransformFilter()
            transform_filter.SetTransform(transform)
            transform_filter.SetInputData(input_data)
            transform_filter.Update()
            mapper.SetInputConnection(transform_filter.GetOutputPort())
            mapper.Update()
        actor.Modified()
