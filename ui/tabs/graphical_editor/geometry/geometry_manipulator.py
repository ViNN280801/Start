from .vtk_geometry import VTKGeometryManipulator
from .gmsh_geometry import GMSHGeometryManipulator
from logger.internal_logger import InternalLogger


class GeometryManipulator:
    
    @staticmethod
    def move(actor, dimtags, x_offset, y_offset, z_offset):
        VTKGeometryManipulator.move(actor, x_offset, y_offset, z_offset)
        GMSHGeometryManipulator.move(dimtags, x_offset, y_offset, z_offset)

    @staticmethod
    def rotate(actor, dimtags, angle_x: float, angle_y: float, angle_z: float):
        VTKGeometryManipulator.rotate(actor, angle_x, angle_y, angle_z)
        GMSHGeometryManipulator.rotate(dimtags, angle_x, angle_y, angle_z)

    @staticmethod
    def scale(actor, dimtags, x_scale: float, y_scale: float, z_scale: float):
        VTKGeometryManipulator.scale(actor, x_scale, y_scale, z_scale)
        GMSHGeometryManipulator.scale(dimtags, x_scale, y_scale, z_scale)

    @staticmethod
    def subtract(first_actor, second_actor, first_dimtags, second_dimtags):
        out_actor = VTKGeometryManipulator.subtract(first_actor, second_actor)
        out_dimtags = GMSHGeometryManipulator.subtract(first_dimtags, second_dimtags)
        return out_actor, out_dimtags
        
    @staticmethod
    def combine(first_actor, second_actor, first_dimtags, second_dimtags):
        out_actor = VTKGeometryManipulator.combine(first_actor, second_actor)
        out_dimtags = GMSHGeometryManipulator.combine(first_dimtags, second_dimtags)
        return out_actor, out_dimtags
    
    @staticmethod
    def intersect(first_actor, second_actor, first_dimtags, second_dimtags):
        out_actor = VTKGeometryManipulator.intersect(first_actor, second_actor)
        out_dimtags = GMSHGeometryManipulator.intersect(first_dimtags, second_dimtags)
        return out_actor, out_dimtags

    @staticmethod
    def cross_section(actor, dimtags, axis, level, angle, size=1e9):
        out_actors = VTKGeometryManipulator.cross_section(actor, axis, level, angle)
        out_dimtags = GMSHGeometryManipulator.cross_section(dimtags, axis, level, angle, size)
        
        if not out_actors or not out_dimtags:
            raise ValueError(f"{InternalLogger.pretty_function_details()}: Failed to create cross section")
        
        return out_actors, out_dimtags
