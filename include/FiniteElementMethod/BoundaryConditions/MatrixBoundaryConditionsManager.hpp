#ifndef MATRIX_BOUNDARY_CONDITIONS_MANAGER_HPP
#define MATRIX_BOUNDARY_CONDITIONS_MANAGER_HPP

#include <map>

#include "FiniteElementMethod/FEMTypes.hpp"

class MatrixBoundaryConditionsManager
{
public:
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

#endif // !MATRIX_BOUNDARY_CONDITIONS_MANAGER_HPP
