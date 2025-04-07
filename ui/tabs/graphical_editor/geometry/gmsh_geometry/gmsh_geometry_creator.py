from gmsh import model
from util.gmsh_helpers import check_tag, check_dimtags, complete_dimtag
from util.util import can_create_plane
from numpy import array
from numpy.linalg import norm
from sys import stderr
from tabs.graphical_editor.geometry.point import Point
from tabs.graphical_editor.geometry.line import Line
from tabs.graphical_editor.geometry.surface import Surface
from tabs.graphical_editor.geometry.sphere import Sphere
from tabs.graphical_editor.geometry.box import Box
from tabs.graphical_editor.geometry.cone import Cone
from tabs.graphical_editor.geometry.cylinder import Cylinder


class GMSHGeometryCreator:
    @staticmethod
    def remove(dimtags):
        """
        Remove specified entities in Gmsh by their dimension and tags.

        Parameters:
        dimtags (list of tuple): List of (dim, tag) tuples where dim is the dimension (e.g., 0 for points, 1 for curves)
                                 and tag is the identifier of the entity to be removed.
        """
        check_dimtags(dimtags)
        model.occ.remove(dimtags)
        model.occ.synchronize()

    @staticmethod
    def create_point(point: Point) -> int:
        """
        Creates the point using
        """
        try:
            tag = model.occ.addPoint(point.x, point.y, point.z)
            check_tag(tag)
            model.occ.synchronize()
        except Exception as e:
            print(
                f"An error occurred while creating the point with Gmsh: {e}",
                file=stderr,
            )
        finally:
            return tag

    @staticmethod
    def create_line(line: Line) -> list:
        """
        Creates the line using
        """
        linetags = []
        try:
            for idx, (x, y, z) in enumerate(line.points, start=1):
                tag = model.occ.addPoint(x, y, z)
                check_tag(tag)
            for i in range(len(line.points) - 1):
                tag = model.occ.addLine(i + 1, i + 2)
                check_tag(tag)
                linetags.append(tag)
            model.occ.synchronize()
        except Exception as e:
            print(
                f"An error occurred while creating the line with Gmsh: {e}", file=stderr
            )
        finally:
            return linetags

    @staticmethod
    def create_surface(surface: Surface) -> int:
        """
        Creates the surface using
        """
        try:
            surface_tag = -1

            for idx, (x, y, z) in enumerate(surface.points, start=1):
                tag = model.occ.addPoint(x, y, z)
                check_tag(tag)
            for i in range(len(surface.points)):
                tag = model.occ.addLine(i + 1, ((i + 1) % len(surface.points)) + 1)
                check_tag(tag)
            loop = model.occ.addCurveLoop(list(range(1, len(surface.points) + 1)))
            check_tag(loop)
            surface_tag = model.occ.addPlaneSurface([loop])
            check_tag(surface_tag)
            model.occ.synchronize()
        except Exception as e:
            print(
                f"An error occurred while creating the surface with Gmsh: {e}",
                file=stderr,
            )
        finally:
            return surface_tag

    @staticmethod
    def create_plane(p1, p2, axis="z"):
        """
        Create a plane from two points and extrude it slightly along the given axis.

        Parameters:
        p1 (list or array): The first point defining the plane in 3D space.
        p2 (list or array): The second point defining the plane in 3D space.
        axis (str): The axis along which to extrude the plane ('x', 'y', 'z'). Default is 'z'.

        Returns:
        list of tuples: The dimension tags of the created plane.

        Raises:
        ValueError: If the points p1 and p2 are identical or the axis is invalid.
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
            raise ValueError("Selected axis must be 'x', 'y', or 'z'.")

        can_create_plane(p1, p2)

        p1 = array(p1)
        p2 = array(p2)
        normal = p2 - p1
        normal = normal / norm(normal)

        plane_size = 1e9
        plane_points = [
            p1 - plane_size,
            [p1[0] + plane_size, p1[1] - plane_size, p1[2] + plane_size],
            [p1[0] + plane_size, p1[1] + plane_size, p1[2] + plane_size],
            [p1[0] - plane_size, p1[1] + plane_size, p1[2] - plane_size],
        ]

        point_tags = [model.occ.addPoint(*point) for point in plane_points]
        for point_tag in point_tags:
            check_tag(point_tag)

        line_tags = [
            model.occ.addLine(point_tags[i], point_tags[(i + 1) % 4]) for i in range(4)
        ]
        for line_tag in line_tags:
            check_tag(line_tag)

        cl_tag = model.occ.addCurveLoop(line_tags)
        check_tag(cl_tag)
        ps_tag = model.occ.addPlaneSurface([cl_tag])
        check_tag(ps_tag)
        model.occ.synchronize()
        ps_dimtag = complete_dimtag("surface", ps_tag)

        extrusion_thickness = 1e-9
        if axis.lower() == "x":
            extrusion_direction = [extrusion_thickness, 0, 0]
        elif axis.lower() == "y":
            extrusion_direction = [0, extrusion_thickness, 0]
        elif axis.lower() == "z":
            extrusion_direction = [0, 0, extrusion_thickness]

        volume_dimtag = model.occ.extrude(ps_dimtag, *extrusion_direction)
        tag = volume_dimtag[1][1]
        check_tag(tag)
        model.occ.synchronize()

        return tag

    @staticmethod
    def create_sphere(sphere: Sphere) -> int:
        """
        Creates the sphere using
        """
        try:
            tag = model.occ.addSphere(sphere.x, sphere.y, sphere.z, sphere.radius)
            check_tag(tag)
            model.occ.synchronize()
        except Exception as e:
            print(
                f"An error occurred while creating the sphere with Gmsh: {e}",
                file=stderr,
            )
        finally:
            return tag

    @staticmethod
    def create_box(box: Box) -> int:
        """
        Creates the box using
        """
        try:
            tag = model.occ.addBox(
                box.x, box.y, box.z, box.length, box.width, box.height
            )
            check_tag(tag)
            model.occ.synchronize()
        except Exception as e:
            print(
                f"An error occurred while creating the box with Gmsh: {e}", file=stderr
            )
        finally:
            return tag

    @staticmethod
    def create_cone(cone: Cone) -> int:
        """
        Creates the cone using
        """
        try:
            # By default make full cone, without 2nd radius
            tag = model.occ.addCone(
                cone.x, cone.y, cone.z, cone.dx, cone.dy, cone.dz, cone.r, 0
            )
            check_tag(tag)
            model.occ.synchronize()
        except Exception as e:
            print(
                f"An error occurred while creating the cone with Gmsh: {e}", file=stderr
            )
        finally:
            return tag

    @staticmethod
    def create_cylinder(cylinder: Cylinder) -> int:
        """
        Creates the cylinder using
        """
        try:
            tag = model.occ.addCylinder(
                cylinder.x,
                cylinder.y,
                cylinder.z,
                cylinder.dx,
                cylinder.dy,
                cylinder.dz,
                cylinder.radius,
            )
            check_tag(tag)
            model.occ.synchronize()
        except Exception as e:
            print(
                f"An error occurred while creating the cylinder with Gmsh: {e}",
                file=stderr,
            )
        finally:
            return tag
