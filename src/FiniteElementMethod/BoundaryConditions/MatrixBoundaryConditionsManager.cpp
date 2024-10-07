#include "FiniteElementMethod/BoundaryConditions/MatrixBoundaryConditionsManager.hpp"
#include "Utilities/Utilities.hpp"

void _setBoundaryConditionForNode(Teuchos::RCP<TpetraMatrixType> matrix, LocalOrdinal nodeID, Scalar value)
{
    if (!matrix->getRowMap()->isNodeGlobalElement(nodeID))
        return; // 0. Skip nodes not owned by this process.

    // 1. Fetch the number of entries in the global row for nodeID.
    size_t numEntries{matrix->getNumEntriesInGlobalRow(nodeID)};

    // 2. Allocate arrays for indices and values based on the number of entries.
    TpetraMatrixType::nonconst_global_inds_host_view_type indices("ind", numEntries);
    TpetraMatrixType::nonconst_values_host_view_type values("val", numEntries);
    size_t checkNumEntries{};

    // 3. Fetch the current row's structure.
    matrix->getGlobalRowCopy(nodeID, indices, values, checkNumEntries);

    // 4. Ensure we fetched the correct number of entries
    if (checkNumEntries != numEntries)
        throw std::runtime_error("Mismatch in number of entries retrieved from the matrix.");

    // 5. Modify the values array to set the diagonal to 'value' and others to 0
    for (size_t i{}; i < numEntries; i++)
        values[i] = (indices[i] == nodeID) ? value : 0.0; // Set diagonal value to the value = 1, other - to 0 to correctly solve matrix equation Ax=b.

    // 6. Replace the modified row back into the matrix.
    matrix->replaceGlobalValues(nodeID, indices, values);
}

void MatrixBoundaryConditionsManager::set(Teuchos::RCP<TpetraMatrixType> matrix, short polynom_order, std::map<GlobalOrdinal, Scalar> const &boundary_conditions)
{
    // Checking types.
    static_assert(std::is_integral_v<GlobalOrdinal>, "GlobalOrdinal must be an integral type.");
    static_assert(std::is_floating_point_v<Scalar>, "Scalar must be a floating point type.");
    static_assert(std::is_integral_v<short>, "Polynomial order must be an integral type.");

    // Check if matrix is valid and initialized.
    if (!matrix || matrix->getGlobalNumEntries() == 0)
    {
        ERRMSG("Matrix is uninitialized or empty.");
        return;
    }

    // Check if boundary conditions are provided.
    if (boundary_conditions.empty())
    {
        WARNINGMSG("Boundary conditions are empty, check them, maybe you forgot to fill them.");
        return;
    }

    try
    {
        // 1. Ensure the matrix is in a state that allows adding or replacing entries.
        matrix->resumeFill();

        // 2. Setting boundary conditions to global stiffness matrix:
        for (auto const &[nodeInGmsh, value] : boundary_conditions)
        {
            for (int j{}; j < polynom_order; ++j)
            {
                GlobalOrdinal nodeID{(nodeInGmsh - 1) * polynom_order + j};

                if (nodeID >= static_cast<GlobalOrdinal>(matrix->getGlobalNumRows()))
                    throw std::out_of_range(util::stringify("Boundary condition refers to node index ",
                                                            nodeID,
                                                            ", which exceeds the maximum row index of ",
                                                            matrix->getGlobalNumRows() - 1, "."));

                _setBoundaryConditionForNode(matrix, nodeID, 1); // There is need to be 1, to have value = `value` in `x` vector, while solve Ax=b.
            }
        }

        // 4. Finilizing filling of the global stiffness matrix.
        matrix->fillComplete();
    }
    catch (std::exception const &ex)
    {
        ERRMSG(ex.what());
        throw;
    }
    catch (...)
    {
        ERRMSG("Unknown error was occured while trying to apply boundary conditions on global stiffness matrix");
        throw;
    }
}
