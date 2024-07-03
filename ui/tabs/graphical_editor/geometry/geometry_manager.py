from vtk import vtkActor

from tabs.graphical_editor.geometry.point import Point
from tabs.graphical_editor.geometry.line import Line
from tabs.graphical_editor.geometry.surface import Surface
from tabs.graphical_editor.geometry.sphere import Sphere
from tabs.graphical_editor.geometry.box import Box
from tabs.graphical_editor.geometry.cone import Cone
from tabs.graphical_editor.geometry.cylinder import Cylinder

from .geometry_creator import GeometryCreator
from .geometry_manipulator import GeometryManipulator

from util.gmsh_helpers import complete_dimtag, complete_dimtags


class GeometryManager:
    # List of tuples: (obj, actor, dimtags)
    # `obj` - string that represents object. Each of classes: Point, Line, Surface, Sphere, Box, Cone, Cylinder has own string representation
    # `actor` - vtkActor object. Aim: To spawn it on scene and perform operations
    # `dimtags` - list of tuples. gmsh standard - [(<dim1>, <tag1>), (<dim2>, <tag2>), ..., (<dimn>, <tagn>)]. Aim: to spawn object in GMSH session and perform operations
    #             Where dim - dimension, tag - ID of the geometry object:
    #             Dims:
    #               0 - point
    #               1 - line
    #               2 - surface
    #               3 - volume
    geometries = []
    
    @staticmethod
    def get_str_by_actor(search_actor: vtkActor) -> str:
        for obj, actor, dimtags in GeometryManager.geometries:
            if actor == search_actor:
                return obj
        raise ValueError(f"There is no such element for the actor: {search_actor}")
    
    @staticmethod
    def get_dimtags_by_actor(search_actor: vtkActor):
        for obj, actor, dimtags in GeometryManager.geometries:
            if actor == search_actor:
                return dimtags
        raise ValueError(f"There is no dimtags for the actor: {search_actor}")
    
    @staticmethod
    def get_actor_by_dimtags(search_dimtags) -> vtkActor:
        for obj, actor, dimtags in GeometryManager.geometries:
            if dimtags == search_dimtags:
                return actor
        raise ValueError(f"There is no actor for these dimtags: {search_dimtags}")
    
    @staticmethod
    def delete_by_actor(search_actor: vtkActor):
        """
        Delete an element from the geometries list by actor.

        Args:
        actor (vtkActor): The actor to delete from the geometries list.

        Raises:
        ValueError: If the actor is not found in the geometries list.
        """
        for index, (obj, actor, dimtags) in enumerate(GeometryManager.geometries):
            if actor == search_actor:
                del GeometryManager.geometries[index]
                return
        raise ValueError(f"Actor {search_actor} not found in the geometries list.")
    
    @staticmethod
    def delete_by_dimtags(search_dimtags):
        """
        Delete an element from the geometries list by dimtags.

        Args:
        dimtags (list of tuples): The dimtags to delete from the geometries list.

        Raises:
        ValueError: If the dimtags are not found in the geometries list.
        """
        for index, (obj, actor, dimtags) in enumerate(GeometryManager.geometries):
            if dimtags == search_dimtags:
                del GeometryManager.geometries[index]
                return
        raise ValueError("Dimtags not found in the geometries list.")
    
    @staticmethod
    def add(obj: str, actor: vtkActor, dimtags):
        GeometryManager.geometries.append((obj, actor, dimtags))
        
    @staticmethod
    def clear():
        GeometryManager.geometries.clear()
    
    @staticmethod
    def empty() -> bool:
        return not GeometryManager.geometries
    
    @staticmethod
    def create_helper(obj_type, object, actor, tag) -> vtkActor:
        from logger.internal_logger import InternalLogger
        
        # 1. Getting string data of the created object
        data = object.__repr__()
        
        # 2. Completing tag to dimtag. Prev: tag(int) -> Current: [(<dim>, <tag>)]
        if obj_type == 'point':
            dimtags = complete_dimtag('point', tag)
        elif obj_type == 'line':
            if not isinstance(tag, list):
                raise TypeError(f"{InternalLogger.pretty_function_details()}: For the line type variable `tag` must be `list`")
            dimtags = complete_dimtags('line', tag)
        elif obj_type == 'surface':
            dimtags = complete_dimtag('surface', tag)
        elif obj_type == 'sphere':
            dimtags = complete_dimtag('sphere', tag)
        elif obj_type == 'box':
            dimtags = complete_dimtag('box', tag)
        elif obj_type == 'cone':
            dimtags = complete_dimtag('cone', tag)
        elif obj_type == 'cylinder':
            dimtags = complete_dimtag('cylinder', tag)
        else:
            raise ValueError(f"{InternalLogger.pretty_function_details()} doesn't have object type [{obj_type}]")
        
        # 3. Adding corresponding data to the internal storage
        GeometryManager.add(data, actor, dimtags)
        
        # 4. Returning result actor
        return actor
    
    @staticmethod
    def create_point(point: Point) -> vtkActor:
        actor, tag = GeometryCreator.create_point(point)
        return GeometryManager.create_helper('point', point, actor, tag)
    
    @staticmethod
    def create_line(line: Line) -> vtkActor:
        actor, tags = GeometryCreator.create_line(line)
        return GeometryManager.create_helper('line', line, actor, tags)
    
    @staticmethod
    def create_surface(surface: Surface) -> vtkActor:
        actor, tag = GeometryCreator.create_surface(surface)
        return GeometryManager.create_helper('surface', surface, actor, tag)
    
    @staticmethod
    def create_sphere(sphere: Sphere) -> vtkActor:
        actor, tag = GeometryCreator.create_sphere(sphere)
        return GeometryManager.create_helper('sphere', sphere, actor, tag)

    @staticmethod
    def create_box(box: Box) -> vtkActor:
        actor, tag = GeometryCreator.create_box(box)
        return GeometryManager.create_helper('box', box, actor, tag)
        
    @staticmethod
    def create_cone(cone: Cone) -> vtkActor:
        actor, tag = GeometryCreator.create_cone(cone)
        return GeometryManager.create_helper('cone', cone, actor, tag)
    
    @staticmethod
    def create_cylinder(cylinder: Cylinder) -> vtkActor:
        actor, tag = GeometryCreator.create_cylinder(cylinder)
        return GeometryManager.create_helper('cylinder', cylinder, actor, tag)
    
    @staticmethod
    def move_(actor, dimtags, x_offset, y_offset, z_offset):
        GeometryManipulator.move(actor, dimtags, x_offset, y_offset, z_offset)
    
    @staticmethod
    def rotate_(actor, dimtags, angle_x: float, angle_y: float, angle_z: float):
        GeometryManipulator.rotate(actor, dimtags, angle_x, angle_y, angle_z)
    
    @staticmethod
    def scale_(actor, dimtags, x_scale: float, y_scale: float, z_scale: float):
        GeometryManipulator.scale(actor, dimtags, x_scale, y_scale, z_scale)
    
    @staticmethod
    def subtract_(first_actor, second_actor, first_dimtags, second_dimtags):
        return GeometryManipulator.subtract(first_actor, second_actor, first_dimtags, second_dimtags)
    
    @staticmethod
    def combine_(first_actor, second_actor, first_dimtags, second_dimtags):
        return GeometryManipulator.combine(first_actor, second_actor, first_dimtags, second_dimtags)
    
    @staticmethod
    def intersect_(first_actor, second_actor, first_dimtags, second_dimtags):
        return GeometryManipulator.intersect(first_actor, second_actor, first_dimtags, second_dimtags)
    
    @staticmethod
    def section_(actor, dimtags, axis, level, size):
        return GeometryManipulator.section(actor, dimtags, axis, level, size)
    
    @staticmethod
    def move(actor: vtkActor, x_offset: float, y_offset: float, z_offset: float):
        dimtags = GeometryManager.get_dimtags_by_actor(actor)
        GeometryManipulator.move(actor, dimtags, x_offset, y_offset, z_offset)
    
    @staticmethod
    def rotate(actor: vtkActor, angle_x: float, angle_y: float, angle_z: float):
        dimtags = GeometryManager.get_dimtags_by_actor(actor)
        GeometryManipulator.rotate(actor, dimtags, angle_x, angle_y, angle_z)
    
    @staticmethod
    def scale(actor: vtkActor, x_scale: float, y_scale: float, z_scale: float):
        dimtags = GeometryManager.get_dimtags_by_actor(actor)
        GeometryManipulator.scale(actor, dimtags, x_scale, y_scale, z_scale)
    
    @staticmethod
    def operation_helper(operation: str, first_actor: vtkActor, second_actor: vtkActor) -> vtkActor:
        from logger.internal_logger import InternalLogger
        
        try:
            # 1. Getting dimtags by their actors
            first_dimtags = GeometryManager.get_dimtags_by_actor(first_actor)
            second_dimtags = GeometryManager.get_dimtags_by_actor(second_actor)
            
            # 2. Do operation
            if operation == 'subtract':
                out_actor, out_dimtags = GeometryManipulator.subtract(first_actor, second_actor, first_dimtags, second_dimtags)
                prefix = 'subtracted'
            elif operation == 'combine':
                out_actor, out_dimtags = GeometryManipulator.combine(first_actor, second_actor, first_dimtags, second_dimtags)
                prefix = 'combined'
            elif operation == 'intersect':
                out_actor, out_dimtags = GeometryManipulator.intersect(first_actor, second_actor, first_dimtags, second_dimtags)
                prefix = 'intersected'
            else:
                raise ValueError(f"{InternalLogger.pretty_function_details()} doesn't support operation [{operation}]")
            
            # 3. Composing string to represent new data
            first_data = GeometryManager.get_str_by_actor(first_actor)
            second_data = GeometryManager.get_str_by_actor(second_actor)
            out_data = f'{prefix}_{first_data}_{second_data}'
            
            # 4. Deleting elements of the previous actors-dimtags from the internal storage by actor
            GeometryManager.delete_by_actor(first_actor)
            GeometryManager.delete_by_actor(second_actor)
            
            # 5. Adding new element
            GeometryManager.add(out_data, out_actor, out_dimtags)
            
            return out_actor
        
        except Exception as e:
            raise ValueError(f"{InternalLogger.pretty_function_details()} can't perform operation {operation}. Reason: {e}")
    
    @staticmethod
    def subtract(first_actor, second_actor) -> vtkActor:
        return GeometryManager.operation_helper('subtract', first_actor, second_actor)
    
    @staticmethod
    def combine(first_actor, second_actor):
        return GeometryManager.operation_helper('combine', first_actor, second_actor)
    
    @staticmethod
    def intersect(first_actor, second_actor):
        return GeometryManager.operation_helper('intersect', first_actor, second_actor)
    
    @staticmethod
    def section(actor, axis, level, size):
        dimtags = GeometryManager.get_dimtags_by_actor(actor)
        out_actors, out_dimtags = GeometryManipulator.section(actor, dimtags, axis, level, size)
        out_actor1, out_actor2 = out_actors
        out_dimtags1, out_dimtags2 = [[x] for x in out_dimtags]
        
        data = GeometryManager.get_str_by_actor(actor)
        out_data_part1 = f'sectioned_part1_{data}'
        out_data_part2 = f'sectioned_part2_{data}'
        
        GeometryManager.delete_by_actor(actor)
        GeometryManager.add(out_data_part1, out_actor1, out_dimtags1)
        GeometryManager.add(out_data_part2, out_actor2, out_dimtags2)
        
        return out_actor1, out_actor2

    @staticmethod
    def mesh(filename: str, mesh_dim: int, mesh_size: float) -> bool:
        from gmsh import model, option, write, clear
        from util.gmsh_helpers import check_msh_filename, check_mesh_dim, check_mesh_size
        
        check_msh_filename(filename)
        check_mesh_dim(mesh_dim)
        check_mesh_size(mesh_size)
        
        created_objects = [obj for obj, actor, dimtags in GeometryManager.geometries]        
        try:            
            # 1. Meshing all obejcts from the internal storage with specified mesh parameters
            # * Methods `create_<obj>` creates geometries in one gmsh session, so we just need to write the meshes down to the file .msh
            option.setNumber("Mesh.MeshSizeMin", mesh_size)
            option.setNumber("Mesh.MeshSizeMax", mesh_size)
            model.mesh.generate(mesh_dim)
            
            # 2. Writing meshes to the file
            write(filename)
            
            success_output = "Successfully meshed objects:\n" + "\n".join([f"{i+1}) {obj}" for i, obj in enumerate(created_objects)])
            print(f"{success_output}\nMesh written to the file '{filename}'")
            
            # 3. Clearing out internal storage and session from the created objects
            GeometryManager.geometries.clear()
            clear() # From gmsh: Clear all loaded models and post-processing data, and add a new empty
            
            return True
        
        except Exception as e:
            fail_output = "Failed to mesh objects:\n" + "\n".join([f"{i+1}) {obj}" for i, obj in enumerate(created_objects)])
            print(f"{fail_output}\nBy reason: {e}")
            
            return False
