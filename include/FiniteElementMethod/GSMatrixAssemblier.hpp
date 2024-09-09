#ifndef GSMATRIXASSEMBLIER_HPP
#define GSMATRIXASSEMBLIER_HPP

/* ATTENTION: Works well only for the polynom order = 1. */

#include "Cell/CellType.hpp"
#include "Cubature/CubatureManager.hpp"
#include "DataHandling/TetrahedronMeshManager.hpp"
#include "FEMTypes.hpp"
#include "Geometry/Mesh.hpp"

/// @brief This class works only with `TetrahedronMeshManager` singleton object.
class GSMatrixAssemblier final
{
private:
    std::string m_mesh_filename;        ///< Filename of the mesh.
    CubatureManager m_cubature_manager; ///< Cubature manager (manages cubature points and weights based on cell type).

    CellType m_cell_type;              ///< Type of the mesh cell (for example: tetrahedron, hexahedron, etc.).
    unsigned short m_desired_accuracy; ///< Desired accuracy for cubature calculation, defining the number of cubature points. Higher values increase precision.
    unsigned short m_polynom_order;    ///< Polynomial order used in the basis functions for the finite element method. Determines the degree of approximation.

    Teuchos::RCP<MapType> m_map;               ///< A smart pointer managing the lifetime of a Map object, which defines the layout of distributed data across the processes in a parallel computation.
    Teuchos::RCP<TpetraMatrixType> m_gsmatrix; ///< Smart pointer on the global stiffness matrix.

    struct MatrixEntry
    {
        GlobalOrdinal row; ///< Global row index for the matrix entry.
        GlobalOrdinal col; ///< Global column index for the matrix entry.
        Scalar value;      ///< Value to be inserted at (row, col) in the global matrix.
    };

    /**
     * @brief Retrieves the vertices of all tetrahedrons in the mesh.
     *
     * This function extracts the coordinates of the vertices for each tetrahedron
     * in the mesh and stores them in a multi-dimensional array (DynRankView).
     * The dimensions of the array are [number of tetrahedrons] x [4 vertices] x [3 coordinates (x, y, z)].
     *
     * @return DynRankView A multi-dimensional array containing the vertices of all tetrahedrons.
     *                     Each tetrahedron is represented by its four vertices, and each vertex has three coordinates (x, y, z).
     * @throw std::runtime_error if an error occurs during the extraction of vertices.
     */
    DynRankView _getTetrahedronVertices();

    /**
     * @brief Computes the local stiffness matrix for a given set of basis gradients and cubature weights.
     * @return The local stiffness matrix.
     */
    DynRankView _computeLocalStiffnessMatrices();

    /**
     * @brief Retrieves matrix entries from calculated local stiffness matrices.
     * @return A vector of matrix entries, each containing global row, column, and value.
     */
    std::vector<GSMatrixAssemblier::MatrixEntry> _getMatrixEntries();

    /**
     * @brief Assemlies global stiffness matrix from the GMSH mesh file (.msh).
     * @return Sparse matrix: global stiffness matrix of the tetrahedron mesh.
     */
    void _assembleGlobalStiffnessMatrix();

public:
    GSMatrixAssemblier(std::string_view mesh_filename, CellType cell_type, short desired_calc_accuracy, short polynom_order);
    ~GSMatrixAssemblier() {}

    /* === Getters for matrix params. === */
    constexpr Teuchos::RCP<TpetraMatrixType> const &getGlobalStiffnessMatrix() const { return m_gsmatrix; }
    size_t rows() const { return m_gsmatrix->getGlobalNumRows(); }
    size_t cols() const { return m_gsmatrix->getGlobalNumCols(); }
    auto &getMeshComponents() { return TetrahedronMeshManager::getInstance(m_mesh_filename.data()); }
    auto const &getMeshComponents() const { return TetrahedronMeshManager::getInstance(m_mesh_filename.data()); }

    /// @brief Checks is the global stiffness matrix empty or not.
    bool empty() const;

    /// &&& Getters. &&& ///
    constexpr CellType getCellType() const { return m_cell_type; }
    constexpr unsigned short getDesiredCalculationAccuracy() const { return m_desired_accuracy; }
    constexpr unsigned short getPolynomOrder() const { return m_polynom_order; }
};

#endif // !GSMATRIXASSEMBLIER_HPP
