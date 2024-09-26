#ifndef GSMASSEMBLIER_HPP
#define GSMASSEMBLIER_HPP

/* ATTENTION: Works well only for the polynom order = 1. */

#include "DataHandling/TetrahedronMeshManager.hpp"
#include "FiniteElementMethod/Cell/CellType.hpp"
#include "FiniteElementMethod/Cubature/CubatureManager.hpp"
#include "FiniteElementMethod/FEMTypes.hpp"
#include "FiniteElementMethod/LinearAlgebraManagers/MatrixManager.hpp"
#include "Geometry/Mesh.hpp"

/// @brief This class works only with `TetrahedronMeshManager` singleton object.
class GSMAssemblier
{
private:
    std::string m_mesh_filename;       ///< Filename of the mesh.
    CellType m_cell_type;              ///< Type of the mesh cell (for example: tetrahedron, hexahedron, etc.).
    unsigned short m_desired_accuracy; ///< Desired accuracy for cubature calculation, defining the number of cubature points. Higher values increase precision.
    unsigned short m_polynom_order;    ///< Polynomial order used in the basis functions for the finite element method. Determines the degree of approximation.

    CubatureManager m_cubature_manager; ///< Cubature manager (manages cubature points and weights based on cell type).
    MatrixManager m_matrix_manager;     ///< Matrix manager (manages initialization and filling of matrices).

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
     * @brief Retrieves matrix entries from calculated local stiffness matrices.
     * @return A vector of matrix entries, each containing global row, column, and value.
     */
    std::vector<MatrixEntry> _getMatrixEntries();

    /**
     * @brief Computes the local stiffness matrix for a given set of basis gradients and cubature weights.
     * @return The local stiffness matrix.
     */
    DynRankView _computeLocalStiffnessMatrices();

    /**
     * @brief Assemlies global stiffness matrix from the GMSH mesh file (.msh).
     * @return Sparse matrix: global stiffness matrix of the tetrahedron mesh.
     */
    void _assembleGlobalStiffnessMatrix();

public:
    GSMAssemblier(std::string_view mesh_filename, CellType cell_type, short desired_calc_accuracy, short polynom_order);
    ~GSMAssemblier() {}

    /* === Getters for matrix params. === */
    auto &getMeshManager() { return TetrahedronMeshManager::getInstance(m_mesh_filename.data()); }
    auto const &getMeshManager() const { return TetrahedronMeshManager::getInstance(m_mesh_filename.data()); }

    /// &&& Getters. &&& ///
    auto getGlobalStiffnessMatrix() { return m_matrix_manager.get(); }
    size_t getRows() const { return m_matrix_manager.rows(); }
    constexpr CellType getCellType() const { return m_cell_type; }
    constexpr unsigned short getDesiredCalculationAccuracy() const { return m_desired_accuracy; }
    constexpr unsigned short getPolynomOrder() const { return m_polynom_order; }
};

#endif // !GSMASSEMBLIER_HPP
