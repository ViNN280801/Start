from vtk import vtkTransform, vtkTransformFilter, vtkActor, vtkBooleanOperationPolyDataFilter
from util import object_operation_executor_helper
from util.vtk_helpers import create_cutting_plane, cut_actor
from sys import stderr
from logger.internal_logger import InternalLogger


class VTKGeometryManipulator:
    
    @staticmethod
    def apply_transformation(actor: vtkActor, transform: vtkTransform):
        """
        Applies a given VTK transformation to a VTK actor.

        This method iterates over the provided actor and applies
        the specified transformation using a vtkTransformFilter.

        Parameters:
        actor (vtkActor): VTK actor to transform.
        transform (vtkTransform): The transformation to apply.

        Raises:
        TypeError: If an item is not a vtkActor.
        """
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
    
    @staticmethod
    def move(actor: vtkActor, x_offset: float, y_offset: float, z_offset: float):
        """
        Translates VTK actor by specified offsets in x, y, and z directions.

        This method creates a translation transform and applies it to the actor.

        Parameters:
        actor (vtkActor): VTK actor to move.
        x_offset (float): Translation offset in the x direction.
        y_offset (float): Translation offset in the y direction.
        z_offset (float): Translation offset in the z direction.

        Raises:
        Exception: If an error occurs during the transformation process.
        """
        try:
            transform = vtkTransform()
            transform.Translate(x_offset, y_offset, z_offset)
            VTKGeometryManipulator.apply_transformation(actor, transform)
        except Exception as e:
            raise RuntimeError(f"{InternalLogger.pretty_function_details()}: Error moving geometry with VTK: {e}", file=stderr)
    
    @staticmethod
    def rotate(actor, angle_x: float, angle_y: float, angle_z: float):
        """
        Rotates VTK actor by specified angles around the x, y, and z axes.

        This method creates a rotation transform and applies it to the actor.

        Parameters:
        actor (vtkActor): VTK actor to rotate.
        angle_x (float): Rotation angle around the x-axis (in degrees).
        angle_y (float): Rotation angle around the y-axis (in degrees).
        angle_z (float): Rotation angle around the z-axis (in degrees).

        Raises:
        Exception: If an error occurs during the transformation process.
        """
        from math import degrees
        try:
            transform = vtkTransform()
            transform.RotateX(degrees(angle_x))
            transform.RotateY(degrees(angle_y))
            transform.RotateZ(degrees(angle_z))
            VTKGeometryManipulator.apply_transformation(actor, transform)
        except Exception as e:
            raise RuntimeError(f"{InternalLogger.pretty_function_details()}: Error rotating geometry with VTK: {e}", file=stderr)

    @staticmethod
    def scale(actor, x_scale: float, y_scale: float, z_scale: float):
        """
        Scales VTK actor uniformly by the specified scale factor.

        This method creates a scaling transform and applies it to the actor.

        Parameters:
        actor (vtkActor): VTK actor to scale.
        x_scale (float): The scale factor by X-axis to apply.
        y_scale (float): The scale factor by Y-axis to apply.
        z_scale (float): The scale factor by Z-axis to apply.

        Raises:
        Exception: If an error occurs during the transformation process.
        """
        try:
            transform = vtkTransform()
            transform.Scale(x_scale, y_scale, z_scale)
            VTKGeometryManipulator.apply_transformation(actor, transform)
        except Exception as e:
            raise RuntimeError(f"{InternalLogger.pretty_function_details()}: Error scaling geometry with VTK: {e}", file=stderr)
    
    @staticmethod
    def subtract(obj_from: vtkActor, obj_to: vtkActor) -> vtkActor:
        booleanOperation = vtkBooleanOperationPolyDataFilter()
        booleanOperation.SetOperationToDifference()
        return object_operation_executor_helper(obj_from, obj_to, booleanOperation)

    @staticmethod
    def combine(obj_from: vtkActor, obj_to: vtkActor) -> vtkActor:
        booleanOperation = vtkBooleanOperationPolyDataFilter()
        booleanOperation.SetOperationToUnion()
        return object_operation_executor_helper(obj_from, obj_to, booleanOperation)

    @staticmethod
    def intersect(obj_from: vtkActor, obj_to: vtkActor) -> vtkActor:
        booleanOperation = vtkBooleanOperationPolyDataFilter()
        booleanOperation.SetOperationToIntersection()
        return object_operation_executor_helper(obj_from, obj_to, booleanOperation)
    
    @staticmethod
    def cross_section(actor: vtkActor, axis: str, level: float, angle: float):
        plane = create_cutting_plane(axis, level, angle)
        out_actors = cut_actor(actor, plane)
        
        if not out_actors or len(out_actors) != 2:
            raise RuntimeError(f"{InternalLogger.pretty_function_details()}: Failed to create cross section in VTK")
        
        return out_actors
