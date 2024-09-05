#include "FEMTypes.hpp"

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
