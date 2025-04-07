import unittest
from unittest.mock import patch
from util.gmsh_helpers import *


class GMSHHelpersTests(unittest.TestCase):
    @patch("gmsh.model.occ.importShapes")
    @patch("gmsh.model.occ.synchronize")
    @patch("gmsh.option.setNumber")
    @patch("gmsh.write")
    @patch("util.check_path_access")
    def test_convert_stp_to_msh(
        self,
        mock_check_path_access,
        mock_write,
        mock_set_number,
        mock_synchronize,
        mock_import_shapes,
    ):
        filename = "test.stp"
        mesh_size = 1.0
        mesh_dim = 3
        output_file = convert_stp_to_msh(filename, mesh_size, mesh_dim)
        mock_import_shapes.assert_called_once_with(filename)
        mock_synchronize.assert_called_once()
        mock_set_number.assert_any_call("Mesh.MeshSizeMin", mesh_size)
        mock_set_number.assert_any_call("Mesh.MeshSizeMax", mesh_size)
        mock_write.assert_called_once_with("test.msh")
        self.assertEqual(output_file, "test.msh")

    def test_check_msh_filename_with_valid_filename(self):
        from util import check_path_access

        with patch("util.check_path_access") as mock_check_path_access:
            mock_check_path_access.return_value = True
            check_msh_filename("../../results/box.msh")
            mock_check_path_access.assert_called_once_with("../../results/box.msh")

    def test_check_msh_filename_with_invalid_extension(self):
        with self.assertRaises(ValueError):
            check_msh_filename("invalid_filename.txt")

    def test_check_msh_filename_with_non_string(self):
        with self.assertRaises(TypeError):
            check_msh_filename(123)

    def test_check_mesh_dim_with_valid_dimension(self):
        check_mesh_dim(2)

    def test_check_mesh_dim_with_invalid_dimension(self):
        with self.assertRaises(ValueError):
            check_mesh_dim(4)

    def test_check_mesh_dim_with_non_integer(self):
        with self.assertRaises(TypeError):
            check_mesh_dim(2.5)

    def test_check_mesh_size_with_valid_size(self):
        check_mesh_size(1.0)

    def test_check_mesh_size_with_invalid_size(self):
        with self.assertRaises(ValueError):
            check_mesh_size(-1.0)

    def test_check_mesh_size_with_non_float(self):
        with self.assertRaises(TypeError):
            check_mesh_size(1)

    def test_check_tag_with_valid_tag(self):
        check_tag(1)

    def test_check_tag_with_invalid_tag(self):
        with self.assertRaises(ValueError):
            check_tag(-1)

    def test_check_dimtags_with_valid_dimtags(self):
        check_dimtags([(3, 1), (3, 2)])

    def test_check_dimtags_with_invalid_dimtags(self):
        with self.assertRaises(ValueError):
            check_dimtags([(2, 1), (3, 2)])

    def test_complete_dimtag_with_valid_geometry(self):
        result = complete_dimtag("box", 1)
        self.assertEqual(result, [(3, 1)])

    def test_complete_dimtag_with_invalid_geometry(self):
        with self.assertRaises(ValueError):
            complete_dimtag("invalid_geometry", 1)

    def test_complete_dimtags_with_valid_geometry(self):
        result = complete_dimtags("sphere", [1, 2, 3])
        self.assertEqual(result, [(3, 1), (3, 2), (3, 3)])

    def test_complete_dimtags_with_invalid_geometry(self):
        with self.assertRaises(ValueError):
            complete_dimtags("invalid_geometry", [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
