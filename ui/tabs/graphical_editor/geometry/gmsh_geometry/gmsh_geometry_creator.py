from gmsh import model
from util import check_tag
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
    def create_point(point: Point) -> int:
        """
        Creates the point using Gmsh.
        """
        try:
            tag = model.occ.addPoint(point.x, point.y, point.z)
            check_tag(tag)
            model.occ.synchronize()
        except Exception as e:
            print(f"An error occurred while creating the point with Gmsh: {e}", file=stderr)
        finally:
            return tag
    
    @staticmethod
    def create_line(line: Line) -> list:
        """
        Creates the line using Gmsh.
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
            print(f"An error occurred while creating the line with Gmsh: {e}", file=stderr)
        finally:
            return linetags
            
    @staticmethod
    def create_surface(surface: Surface) -> int:
        """
        Creates the surface using Gmsh.
        """
        try:
            for idx, (x, y, z) in enumerate(surface.points, start=1):
                tag = model.occ.addPoint(x, y, z, tag=idx)
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
            print(f"An error occurred while creating the surface with Gmsh: {e}", file=stderr)
        finally:
            return surface_tag
    
    @staticmethod
    def create_sphere(sphere: Sphere) -> int:
        """
        Creates the sphere using Gmsh.
        """
        try:
            tag = model.occ.addSphere(sphere.x, sphere.y, sphere.z, sphere.radius)
            check_tag(tag)
            model.occ.synchronize()
        except Exception as e:
            print(f"An error occurred while creating the sphere with Gmsh: {e}", file=stderr)
        finally:
            return tag

    @staticmethod
    def create_box(box: Box) -> int:
        """
        Creates the box using Gmsh.
        """
        try:
            tag = model.occ.addBox(box.x, box.y, box.z, box.length, box.width, box.height)
            check_tag(tag)
            model.occ.synchronize()
        except Exception as e:
            print(f"An error occurred while creating the box with Gmsh: {e}", file=stderr)
        finally:
            return tag
    
    @staticmethod
    def create_cone(cone: Cone) -> int:
        """
        Creates the cone using Gmsh.
        """
        try:
            # By default make full cone, without 2nd radius
            tag = model.occ.addCone(cone.x, cone.y, cone.z, cone.dx, cone.dy, cone.dz, cone.r, 0)
            check_tag(tag)
            model.occ.synchronize()
        except Exception as e:
            print(f"An error occurred while creating the cone with Gmsh: {e}", file=stderr)
        finally:
            return tag
    
    @staticmethod
    def create_cylinder(cylinder: Cylinder) -> int:
        """
        Creates the cylinder using Gmsh.
        """
        try:
            tag = model.occ.addCylinder(cylinder.x, cylinder.y, cylinder.z, cylinder.dx, cylinder.dy, cylinder.dz, cylinder.radius)
            check_tag(tag)
            model.occ.synchronize()
        except Exception as e:
            print(f"An error occurred while creating the cylinder with Gmsh: {e}", file=stderr)
        finally:
            return tag
