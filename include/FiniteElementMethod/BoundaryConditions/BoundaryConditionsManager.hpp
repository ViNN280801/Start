#ifndef BOUNDARY_CONDITIONS_MANAGER_HPP
#define BOUNDARY_CONDITIONS_MANAGER_HPP

#include "FiniteElementMethod/FEMTypes.hpp"

/**
 * @class BoundaryConditionsManager
 * @brief Manages the application of Dirichlet boundary conditions for finite element analysis.
 *
 * This class provides static methods to apply Dirichlet boundary conditions to both the solution vector
 * and the global stiffness matrix in the context of finite element methods (FEM). Dirichlet boundary conditions
 * fix the values of the solution at specific nodes, enforcing conditions like displacement or temperature at
 * certain points in the domain.
 *
 * The class supports different polynomial orders and degrees of freedom (DOFs) at each node. The boundary
 * conditions are specified as a map where the keys represent global node indices (based on GMSH 1-based indexing),
 * and the values specify the fixed values to apply at those nodes. The node indices are adjusted internally to match
 * 0-based indexing for the solution vector and matrix.
 *
 * ### Key Features:
 * - **Solution Vector Adjustment**: Modifies the provided solution vector to fix specific DOF values based on
 *   the given boundary conditions.
 * - **Global Stiffness Matrix Modification**: Alters rows and columns of the global stiffness matrix to correctly
 *   impose Dirichlet boundary conditions, ensuring numerical stability and accuracy.
 *
 * @note The class is designed to handle Dirichlet boundary conditions only. Neumann or mixed boundary conditions
 *       are not handled by this class.
 *
 * The boundary condition map is applied uniformly across different cell types and polynomial orders.
 * The polynomial order defines the number of DOFs per node, which the class uses to correctly index
 * and apply the boundary conditions.
 *
 * ### Error Handling:
 * - Throws a runtime error if an invalid node index is encountered (e.g., out of range of the solution vector or matrix).
 * - Issues a warning if the boundary conditions map is empty.
 *
 * ### Usage Example:
 * @code
 * // Define boundary conditions (e.g., fix node 5 at value 0.0)
 * std::map<GlobalOrdinal, Scalar> boundary_conditions = { {5, 0.0} };
 *
 * // Apply boundary conditions to the solution vector and stiffness matrix
 * BoundaryConditionsManager::set(solution_vector, 2, boundary_conditions);
 * BoundaryConditionsManager::set(global_matrix, 2, boundary_conditions);
 * @endcode
 *
 * @see Teuchos::RCP for reference-counted pointers and Tpetra for matrix and vector types used in parallel FEM solvers.
 */
class BoundaryConditionsManager
{
public:
    /**
     * @brief Applies boundary conditions to the solution vector.
     *
     * This method modifies the provided solution vector by setting values at specific global
     * indices based on the boundary conditions. Each entry in the boundary conditions map
     * specifies a node index (in GMSH 1-based indexing) and the value to assign at that node.
     * The node index is adjusted to match the internal 0-based indexing of the solution vector.
     *
     * @param vector The vector where boundary conditions are applied.
     * @param polynom_order The polynomial order used to determine the number of degrees
     * of freedom (DOFs) per node.
     * @param boundary_conditions A map of node IDs (key) and their respective values (value)
     * for applying boundary conditions.
     *
     * ### Error Handling:
     * - If an invalid node index is encountered (out of range), a runtime error is thrown.
     * - A warning is issued if the boundary conditions map is empty.
     */
    static void set(Teuchos::RCP<TpetraVectorType> vector, short polynom_order, std::map<GlobalOrdinal, Scalar> const &boundary_conditions);

    /**
     * @brief Applies boundary conditions to the global stiffness matrix.
     *
     * This method modifies the matrix based on boundary conditions, ensuring correct values are set
     * to the corresponding rows and columns.
     *
     * @param matrix The matrix to which boundary conditions are applied.
     * @param polynom_order Polynomial order for determining the number of DOFs per node.
     * @param boundary_conditions Map of node IDs and values for boundary conditions.
     */
    static void set(Teuchos::RCP<TpetraMatrixType> matrix, short polynom_order, std::map<GlobalOrdinal, Scalar> const &boundary_conditions);
};

#endif // !BOUNDARY_CONDITIONS_MANAGER_HPP
