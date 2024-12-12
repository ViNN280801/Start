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

from logger.internal_logger import InternalLogger
from util.gmsh_helpers import(
    complete_dimtag, complete_dimtags, check_tag, 
    gmsh_clear, check_msh_filename, check_mesh_dim, 
    check_mesh_size
)
from util.vtk_helpers import cut_actor
from gmsh import model


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
        raise ValueError(f"Failed to delete object by actor: Actor {search_actor} not found in the geometries list.")
    
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
        raise ValueError(f"Failed to delete object by actor: Dimtags {search_dimtags} not found in the geometries list.")
    
    @staticmethod
    def add(obj: str, actor: vtkActor, dimtags):
        GeometryManager.geometries.append((obj, actor, dimtags))
    
    @staticmethod
    def remove(vtkWidget, renderer, actor_to_remove, needResetCamera):
        try:
            dimtags_to_remove = GeometryManager.get_dimtags_by_actor(actor_to_remove)
            GeometryCreator.remove(vtkWidget, renderer, actor_to_remove, needResetCamera, dimtags_to_remove)
            GeometryManager.delete_by_actor(actor_to_remove)

        except Exception as e:
            raise ValueError(f"Failed to remove geometrical object: {e}")
        
    @staticmethod
    def clear():
        GeometryManager.geometries.clear() # 1. Clearing out internal storage of geometries
        gmsh_clear()                       # 2. Clear all loaded models and post-processing data
    
    @staticmethod
    def empty() -> bool:
        return not GeometryManager.geometries
    
    @staticmethod
    def create_helper(obj_type, object, actor, tag) -> vtkActor:
        """
        Helper function to create VTK actors and associate them with corresponding dimension tags.

        Parameters:
        obj_type (str): The type of the object being created (e.g., 'point', 'line', 'surface', 'sphere', 'box', 'cone', 'cylinder').
        object: The VTK object being created.
        actor: The VTK actor associated with the object.
        tag (int or list): The tag or tags associated with the object. Must be a list for 'line' type.

        Returns:
        vtkActor: The resulting VTK actor associated with the object.

        Raises:
        TypeError: If the `tag` is not a list for 'line' type.
        ValueError: If the `obj_type` is not one of the supported types.

        Notes:
        This function performs the following steps:
        1. Retrieves a string representation of the created object.
        2. Converts the tag to dimension tags based on the object type.
        - For 'point', 'surface', 'sphere', 'box', 'cone', and 'cylinder' types, it uses the `complete_dimtag` function.
        - For 'line' type, it checks that the `tag` is a list and then uses the `complete_dimtags` function.
        - If the `obj_type` is not supported, it raises a ValueError.
        3. Adds the object data, actor, and dimension tags to the internal storage using `GeometryManager.add`.
        4. Returns the resulting VTK actor.

        Example:
        create_helper('sphere', sphere_object, sphere_actor, sphere_tag)
        """
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
    def create_plane(p1, p2, axis='z'):
        vtkplane, actor, tag = GeometryCreator.create_plane(p1, p2, axis)
        return vtkplane, GeometryManager.create_helper('box', vtkplane, actor, tag)
    
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
    def transform_general(transform_name: str, actor: vtkActor, x_value: float, y_value: float, z_value: float):
        """
        Perform a transformation on a given actor based on the specified transformation name.

        Parameters:
        transform_name (str): The name of the transformation to be performed. Should be 'move', 'rotate', or 'scale'.
        actor (vtkActor): The actor to be transformed.
        x_value (float): The x-component value for the transformation.
        y_value (float): The y-component value for the transformation.
        z_value (float): The z-component value for the transformation.

        Raises:
        ValueError: If the provided transformation name is not 'move', 'rotate', or 'scale'.
        """
        transform_name = transform_name.lower()
        
        if transform_name == 'move':
            GeometryManager.move(actor, x_value, y_value, z_value)
        elif transform_name == 'rotate':
            GeometryManager.rotate(actor, x_value, y_value, z_value)
        elif transform_name == 'scale':
            GeometryManager.scale(actor, x_value, y_value, z_value)
        else:
            raise ValueError(f"Failed to perform transformation with name {transform_name}. There is no such transformation")
    
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
    def cross_section(actor, point1, point2, axis):
        vtkplane, plane_actor = GeometryManager.create_plane(point1, point2, axis)
        actor_dimtags = GeometryManager.get_dimtags_by_actor(actor)
        plane_dimtags = GeometryManager.get_dimtags_by_actor(plane_actor)
        
        out_actors, out_dimtags = GeometryManipulator.cross_section(actor, vtkplane, actor_dimtags, plane_dimtags)

        if len(out_actors) != 2 or len(out_dimtags) != 2:
            raise ValueError(f"Can't create cross-section. In result got <{len(out_actors)}> visual objects and <{len(out_dimtags)}> geometrical objects")
        
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
    def process_surfaces():
        """
        Process all surfaces and add them to a physical group in Gmsh.
        This method is called within the `mesh` method to ensure that all surfaces
        are properly prepared for meshing.
        """
        from gmsh import model
        
        surface_tags = []
        for _, actor, dimtags in GeometryManager.geometries:
            surface_tags.extend([tag for dim, tag in dimtags if dim == 2])

        if surface_tags:
            model.addPhysicalGroup(2, surface_tags, tag=1)

    @staticmethod
    def mesh(filename: str, mesh_dim: int, mesh_size: float) -> bool:
        from gmsh import option, write, clear
        
        check_msh_filename(filename)
        check_mesh_dim(mesh_dim)
        check_mesh_size(mesh_size)
        
        created_objects = [obj for obj, actor, dimtags in GeometryManager.geometries]
        try:
            # 0. Extending physical group for surface before meshing
            GeometryManager.process_surfaces()
            
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
