from gmsh import model
from sys import stderr
from util import check_dimtags
from util.gmsh_helpers import gmsh_rotate, gmsh_cut_with_plane


class GMSHGeometryManipulator:
    @staticmethod
    def move(dimtags, x_offset: float, y_offset: float, z_offset: float):
        """
        Translates the geometries identified by dimtags by the given offsets.

        Parameters:
        dimtags (list of tuples): A list of (dimension, tag) tuples identifying the geometries to translate.
        x_offset (float): The offset along the x-axis.
        y_offset (float): The offset along the y-axis.
        z_offset (float): The offset along the z-axis.

        Raises:
        ValueError: If any dimension in the dimtags is not equal to 3.
        """
        try:
            check_dimtags(dimtags)
            model.occ.translate(dimtags, x_offset, y_offset, z_offset)
            model.occ.synchronize()
        except Exception as e:
            print(f"Error moving geometry with GMSH: {e}", file=stderr)

    @staticmethod
    def rotate(dimtags, angle_x: float, angle_y: float, angle_z: float):
        gmsh_rotate(dimtags, angle_x, angle_y, angle_z)

    @staticmethod
    def scale(dimtags, x_scale: float, y_scale: float, z_scale: float):
        """
        Scales the geometries identified by dimtags by the given scale factor.

        Parameters:
        dimtags (list of tuples): A list of (dimension, tag) tuples identifying the geometries to scale.
        x_scale (float): The scale factor by X-axis to apply.
        y_scale (float): The scale factor by Y-axis to apply.
        z_scale (float): The scale factor by Z-axis to apply.

        Raises:
        ValueError: If any dimension in the dimtags is not equal to 3.
        """
        try:
            check_dimtags(dimtags)
            for dim, tag in dimtags:
                bbox = model.getBoundingBox(dim, tag)
                center = [(bbox[i + 3] + bbox[i]) / 2 for i in range(3)]
                model.occ.dilate(
                    [(dim, tag)],
                    center[0],
                    center[1],
                    center[2],
                    x_scale,
                    y_scale,
                    z_scale,
                )
                model.occ.synchronize()
        except Exception as e:
            print(f"Error scaling geometry with GMSH: {e}", file=stderr)

    @staticmethod
    def subtract(from_dimtags, what_dimtags):
        try:
            check_dimtags(from_dimtags)
            check_dimtags(what_dimtags)

            out_dimtags, _ = model.occ.cut(from_dimtags, what_dimtags)
            if len(out_dimtags) != 1:
                raise ValueError(
                    f"Subtraction failed between dimtags: {from_dimtags} and {what_dimtags}"
                )
            check_dimtags(out_dimtags)
            model.occ.synchronize()

            return out_dimtags

        except Exception:
            return None

    @staticmethod
    def combine(first_dimtags, second_dimtags):
        try:
            check_dimtags(first_dimtags)
            check_dimtags(second_dimtags)

            out_dimtags, _ = model.occ.fuse(first_dimtags, second_dimtags)
            if len(out_dimtags) != 1:
                raise ValueError(
                    f"Union failed between dimtags: {first_dimtags} and {second_dimtags}"
                )
            check_dimtags(out_dimtags)
            model.occ.synchronize()

            return out_dimtags

        except Exception:
            return None

    @staticmethod
    def intersect(first_dimtags, second_dimtags):
        try:
            check_dimtags(first_dimtags)
            check_dimtags(second_dimtags)

            out_dimtags, _ = model.occ.intersect(first_dimtags, second_dimtags)
            if len(out_dimtags) != 1:
                raise ValueError(
                    f"Intersection failed between dimtags: {first_dimtags} and {second_dimtags}"
                )
            check_dimtags(out_dimtags)
            model.occ.synchronize()

            return out_dimtags

        except Exception:
            return None

    @staticmethod
    def cross_section(input_dimtags, plane_dimtags):
        out_dimtags = gmsh_cut_with_plane(input_dimtags, plane_dimtags)
        check_dimtags(out_dimtags)
        return out_dimtags
