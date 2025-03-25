from .geometry_constants import *


class Sphere:
    """
    A class to represent a sphere geometry.

    Attributes
    ----------
    x : float
        The x-coordinate of the sphere's center.
    y : float
        The y-coordinate of the sphere's center.
    z : float
        The z-coordinate of the sphere's center.
    radius : float
        The radius of the sphere.
    mesh_resolution : int
        The triangle vtkLinearSubdivisionFilter count of the subdivisions.
    phi_resolution : int, optional
        The phi resolution of the sphere (default is DEFAULT_SPHERE_PHI_RESOLUTION).
    theta_resolution : int, optional
        The theta resolution of the sphere (default is DEFAULT_SPHERE_THETA_RESOLUTION).
    """

    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        radius: float,
        mesh_resolution: int,
        phi_resolution: int = DEFAULT_SPHERE_PHI_RESOLUTION,
        theta_resolution: int = DEFAULT_SPHERE_THETA_RESOLUTION,
    ):
        """
        Constructs all the necessary attributes for the sphere object.

        Parameters
        ----------
            x : float
                The x-coordinate of the sphere's center.
            y : float
                The y-coordinate of the sphere's center.
            z : float
                The z-coordinate of the sphere's center.
            radius : float
                The radius of the sphere.
            mesh_resolution : int
                The triangle vtkLinearSubdivisionFilter count of the subdivisions.
            phi_resolution : int, optional
                The phi resolution of the sphere (default is DEFAULT_SPHERE_PHI_RESOLUTION).
            theta_resolution : int, optional
                The theta resolution of the sphere (default is DEFAULT_SPHERE_THETA_RESOLUTION).
        """
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius
        self.mesh_resolution = mesh_resolution
        self.phi_resolution = phi_resolution
        self.theta_resolution = theta_resolution

    def __repr__(self):
        """
        Returns a string representation of the sphere.

        Returns
        -------
        str
            A string representation of the sphere.
        """
        sphere_data_str = []
        sphere_data_str.append(f"Center: ({self.x}, {self.y}, {self.z})")
        sphere_data_str.append(f"Radius: {self.radius}")
        return "\n".join(sphere_data_str)
