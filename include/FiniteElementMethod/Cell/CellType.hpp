#ifndef CELL_TYPE_HPP
#define CELL_TYPE_HPP

/**
 * @enum CellType
 * @brief Enumeration of all supported cell types, including both 2D and 3D mesh cells.
 *
 * This enumeration lists all the cell types that the CellSelector class can manage, covering a wide range of finite element method (FEM) applications.
 */
enum class CellType
{
    // 2D Cells
    Triangle, ///< 2D Triangle with 3 nodes.
    Pentagon, ///< 2D Pentagon with 5 nodes.
    Hexagon,  ///< 2D Hexagon with 6 nodes.

    // 3D Volumetric Cells
    Tetrahedron, ///< 3D Tetrahedron with 4 nodes.
    Pyramid,     ///< 3D Pyramid with 5 nodes.
    Wedge,       ///< 3D Wedge with 6 nodes.
    Hexahedron,  ///< 3D Hexahedron with 8 nodes.
};

#endif // !CELL_TYPE_HPP
