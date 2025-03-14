#ifndef TETRAHEDRON_CELL_HPP
#define TETRAHEDRON_CELL_HPP

#include "Geometry/Basics/BaseTypes.hpp"

/**
 * @brief Represents the parameters of a tetrahedron cell in a 3D mesh.
 *
 * The `TetrahedronCell` structure is used to store the geometric representation of a
 * tetrahedron in a volumetric mesh. Each tetrahedron is uniquely identified by its ID
 * (as a key in `TetrahedronCellMap`) and contains:
 * - The geometric representation of the tetrahedron.
 */
struct TetrahedronCell
{
    Tetrahedron tetrahedron; ///< CGAL::Tetrahedron_3 object representing the tetrahedron.

    TetrahedronCell() = default;
    TetrahedronCell(TetrahedronCell const &) = default;
    TetrahedronCell(TetrahedronCell &&) = default;
    TetrahedronCell &operator=(TetrahedronCell const &) = default;
    TetrahedronCell &operator=(TetrahedronCell &&) = default;
    TetrahedronCell(Tetrahedron_cref tetrahedron_) : tetrahedron(tetrahedron_) {}
};
using TetrahedronCell_ref = TetrahedronCell &;
using TetrahedronCell_cref = TetrahedronCell const &;

/**
 * @brief A map for storing and accessing tetrahedron cells by their unique IDs.
 *
 * The `TetrahedronCellMap` provides an efficient hash-based container for tetrahedron cells.
 * Each entry consists of:
 * - Key: A unique identifier for the tetrahedron (`size_t`).
 * - Value: A `TetrahedronCell` structure containing geometric data.
 *
 * This map is suitable for managing volumetric meshes with efficient access patterns.
 */
using TetrahedronCellMap = std::unordered_map<size_t, TetrahedronCell>;
using TetrahedronCellMap_ref = TetrahedronCellMap &;
using TetrahedronCellMap_cref = TetrahedronCellMap const &;

#endif // !TETRAHEDRON_CELL_HPP
