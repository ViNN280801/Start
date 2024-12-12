#ifndef GSMASSEMBLER_HPP
#define GSMASSEMBLER_HPP

/* ATTENTION: Works well only for the polynom order = 1. */

#include "DataHandling/TetrahedronMeshManager.hpp"
#include "FiniteElementMethod/Cell/CellType.hpp"
#include "FiniteElementMethod/Cubature/CubatureManager.hpp"
#include "FiniteElementMethod/FEMTypes.hpp"
#include "FiniteElementMethod/LinearAlgebraManagers/MatrixManager.hpp"
#include "Geometry/Mesh.hpp"

/**
 * @class GSMAssembler
 * @brief A class for assembling the global stiffness matrix (GSM) in finite element analysis (FEA).
 *
 * The `GSMAssembler` class manages the assembly of a global stiffness matrix from a GMSH mesh file (.msh).
 * It integrates mesh management, cubature management (for numerical integration), and matrix operations
 * to facilitate finite element method (FEM) computations. The class supports operations for tetrahedron meshes
 * and can be extended for other cell types.
 *
 * ### Key Features
 * - **Mesh Handling**: Manages and processes tetrahedron meshes using `TetrahedronMeshManager`.
 * - **Cubature Management**: Determines cubature points and weights based on the desired accuracy and cell type.
 * - **Matrix Assembly**: Computes local stiffness matrices and assembles them into a global stiffness matrix.
 *
 * ### Design Considerations
 * - The class is designed for extensibility, allowing different cell types to be supported.
 * - Uses dynamic rank views (DynRankView) to handle multi-dimensional data arrays for mesh vertices and matrices.
 * - Focuses on modularity by delegating mesh, cubature, and matrix-related tasks to specialized classes.
 *
 * ### Usage
 * - Initialize the class with the mesh filename, cell type, desired accuracy, and polynomial order.
 * - Use provided methods to extract vertices, compute matrices, and assemble the global stiffness matrix.
 */
class GSMAssembler
{
private:
    std::string m_mesh_filename;       ///< Filename of the mesh.
    CellType m_cell_type;              ///< Type of the mesh cell (for example: tetrahedron, hexahedron, etc.).
    unsigned short m_desired_accuracy; ///< Desired accuracy for cubature calculation, defining the number of cubature points. Higher values increase precision.
    unsigned short m_polynom_order;    ///< Polynomial order used in the basis functions for the finite element method. Determines the degree of approximation.

    TetrahedronMeshManager m_meshManager; ///< Manages the mesh and its properties.
    CubatureManager m_cubature_manager;   ///< Cubature manager (manages cubature points and weights based on cell type).
    MatrixManager m_matrix_manager;       ///< Matrix manager (manages initialization and filling of matrices).

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

public:
    /**
     * @brief Constructor for the `GSMAssembler` class.
     *
     * Initializes the assembler with the mesh file, cell type, desired calculation accuracy,
     * and polynomial order. These parameters define the mesh, the degree of numerical integration,
     * and the approximation order for the FEM.
     *
     * @param mesh_filename Filename of the GMSH mesh file (.msh).
     * @param cell_type Type of the cell in the mesh (e.g., tetrahedron, hexahedron).
     * @param desired_calc_accuracy Desired accuracy for cubature calculation (number of points).
     * @param polynom_order Polynomial order for FEM basis functions.
     */
    GSMAssembler(std::string_view mesh_filename, CellType cell_type, short desired_calc_accuracy, short polynom_order);

    /// @brief Dtor.
    ~GSMAssembler() {}

    /**
     * @brief Retrieves the mesh manager instance.
     * @return A reference to the mesh manager.
     */
    auto &getMeshManager() { return m_meshManager; }

    /**
     * @brief Retrieves the mesh manager instance (const version).
     * @return A const reference to the mesh manager.
     */
    auto const &getMeshManager() const { return m_meshManager; }

    /**
     * @brief Retrieves the global stiffness matrix.
     * @return The global stiffness matrix managed by `MatrixManager`.
     */
    auto getGlobalStiffnessMatrix() { return m_matrix_manager.get(); }

    /**
     * @brief Retrieves the number of rows in the global stiffness matrix.
     * @return The number of rows in the global stiffness matrix.
     */
    size_t getRows() const { return m_matrix_manager.rows(); }

    /**
     * @brief Retrieves the cell type of the mesh.
     * @return The cell type of the mesh.
     */
    constexpr CellType getCellType() const { return m_cell_type; }

    /**
     * @brief Retrieves the desired calculation accuracy for cubature.
     * @return The desired cubature accuracy.
     */
    constexpr unsigned short getDesiredCalculationAccuracy() const { return m_desired_accuracy; }

    /**
     * @brief Retrieves the polynomial order for FEM basis functions.
     * @return The polynomial order.
     */
    constexpr unsigned short getPolynomOrder() const { return m_polynom_order; }
};

#endif // !GSMASSEMBLER_HPP
