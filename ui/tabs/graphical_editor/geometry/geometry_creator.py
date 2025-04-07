from tabs.graphical_editor.geometry.point import Point
from tabs.graphical_editor.geometry.line import Line
from tabs.graphical_editor.geometry.surface import Surface
from tabs.graphical_editor.geometry.sphere import Sphere
from tabs.graphical_editor.geometry.box import Box
from tabs.graphical_editor.geometry.cone import Cone
from tabs.graphical_editor.geometry.cylinder import Cylinder

from .vtk_geometry import VTKGeometryCreator
from .gmsh_geometry import GMSHGeometryCreator


class GeometryCreator:
    @staticmethod
    def remove(vtkWidget, renderer, actor, needResetCamera, dimtags):
        VTKGeometryCreator.remove(vtkWidget, renderer, actor, needResetCamera)
        GMSHGeometryCreator.remove(dimtags)

    @staticmethod
    def create_point(point: Point):
        out_actor = VTKGeometryCreator.create_point(point)
        out_tag = GMSHGeometryCreator.create_point(point)
        return out_actor, out_tag

    @staticmethod
    def create_line(line: Line):
        out_actor = VTKGeometryCreator.create_line(line)
        out_tag = GMSHGeometryCreator.create_line(line)
        return out_actor, out_tag

    @staticmethod
    def create_surface(surface: Surface):
        out_actor = VTKGeometryCreator.create_surface(surface)
        out_tag = GMSHGeometryCreator.create_surface(surface)
        return out_actor, out_tag

    @staticmethod
    def create_plane(p1, p2, axis="z"):
        out_vtkplane, out_plane_actor = VTKGeometryCreator.create_plane(p1, p2, axis)
        out_tag = GMSHGeometryCreator.create_plane(p1, p2, axis)
        return out_vtkplane, out_plane_actor, out_tag

    @staticmethod
    def create_sphere(sphere: Sphere):
        out_actor = VTKGeometryCreator.create_sphere(sphere)
        out_tag = GMSHGeometryCreator.create_sphere(sphere)
        return out_actor, out_tag

    @staticmethod
    def create_box(box: Box):
        out_actor = VTKGeometryCreator.create_box(box)
        out_tag = GMSHGeometryCreator.create_box(box)
        return out_actor, out_tag

    @staticmethod
    def create_cone(cone: Cone):
        out_actor = VTKGeometryCreator.create_cone(cone)
        out_tag = GMSHGeometryCreator.create_cone(cone)
        return out_actor, out_tag

    @staticmethod
    def create_cylinder(cylinder: Cylinder):
        out_actor = VTKGeometryCreator.create_cylinder(cylinder)
        out_tag = GMSHGeometryCreator.create_cylinder(cylinder)
        return out_actor, out_tag
