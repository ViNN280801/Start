from vtk import (
    vtkPoints, vtkVertexGlyphFilter,
    vtkPolyLine, vtkCellArray,
    vtkPolygon,
    vtkSphereSource,
    vtkCubeSource,
    vtkConeSource,
    vtkCylinderSource,
    vtkActor, vtkPolyData, vtkPolyDataMapper, vtkTriangleFilter, vtkLinearSubdivisionFilter
)
from sys import stderr
from tabs.graphical_editor.geometry.point import Point
from tabs.graphical_editor.geometry.line import Line
from tabs.graphical_editor.geometry.surface import Surface
from tabs.graphical_editor.geometry.sphere import Sphere
from tabs.graphical_editor.geometry.box import Box
from tabs.graphical_editor.geometry.cone import Cone
from tabs.graphical_editor.geometry.cylinder import Cylinder


class VTKGeometryCreator:
    
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
            print(f"An error occurred while creating the point with VTK: {e}", file=stderr)
            
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
            print(f"An error occurred while creating the line with VTK: {e}", file=stderr)
    
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
            print(f"An error occurred while creating the surface with VTK: {e}", file=stderr)

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
            print(f"An error occurred while creating the sphere with VTK: {e}", file=stderr)
    
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
            print(f"An error occurred while creating the box with VTK: {e}", file=stderr)
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
            cone_source.SetCenter(cone.x, cone.y, cone.z)
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
            print(f"An error occurred while creating the cone with VTK: {e}", file=stderr)
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
            cylinder_source.SetRadius(cylinder.radius)
            cylinder_source.SetHeight(cylinder.dz)
            cylinder_source.SetCenter(cylinder.x, cylinder.y, cylinder.z + cylinder.dz / 2)
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

            return actor
        
        except Exception as e:
            print(f"An error occurred while creating the cylinder with VTK: {e}", file=stderr)
