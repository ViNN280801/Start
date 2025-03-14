#include "FiniteElementMethod/BoundaryConditions/BoundaryConditionsExceptions.hpp"
#include "FiniteElementMethod/BoundaryConditions/VectorBoundaryConditionManager.hpp"
#include "Utilities/Utilities.hpp"

void VectorBoundaryConditionsManager::set(Teuchos::RCP<TpetraVectorType> vector, short polynom_order, std::map<GlobalOrdinal, Scalar> const &boundary_conditions)
{
    // Checking types.
    static_assert(std::is_integral_v<GlobalOrdinal>, "GlobalOrdinal must be an integral type.");
    static_assert(std::is_floating_point_v<Scalar>, "Scalar must be a floating point type.");
    static_assert(std::is_integral_v<short>, "Polynomial order must be an integral type.");

    // Check if vector is valid and initialized.
    if (!vector || vector->getGlobalLength() == 0)
    {
        ERRMSG("Vector is uninitialized or empty.");
        return;
    }

    // Check if boundary conditions are provided.
    if (boundary_conditions.empty())
    {
        WARNINGMSG("Boundary conditions are empty, check them, maybe you forgot to fill them");
        return;
    }

    try
    {
        // Setting boundary conditions to vector:
        for (auto const &[nodeInGmsh, value] : boundary_conditions)
            for (int j{}; j < polynom_order; ++j)
            {
                // -1 because indexing in GMSH is on 1 bigger than in the program.
                GlobalOrdinal nodeID{(nodeInGmsh - 1) * polynom_order + j};

                if (nodeID >= static_cast<GlobalOrdinal>(vector->getGlobalLength()))
                    START_THROW_EXCEPTION(VectorBoundaryConditionsNodeIDOutOfRangeException,
                                          util::stringify("Boundary condition refers to node index ",
                                                          nodeID,
                                                          ", which exceeds the maximum row index of ",
                                                          vector->getGlobalLength() - 1, "."));

                vector->replaceGlobalValue(nodeID, value); // Modifying the RHS vector is necessary to solve the equation Ax=b.
            }
    }
    catch (std::exception const &ex)
    {
        std::string errorMessage{util::stringify("Error was occured while trying to apply boundary conditions on vector (`b`) in equation Ax=b: ", ex.what())};
        ERRMSG(errorMessage);
        START_THROW_EXCEPTION(VectorBoundaryConditionsSettingException, errorMessage);
    }
    catch (...)
    {
        std::string errorMessage{"Unknown error was occured while trying to apply boundary conditions on vector (`b`) in equation Ax=b"};
        ERRMSG(errorMessage);
        START_THROW_EXCEPTION(VectorBoundaryConditionsUnknownException, errorMessage);
    }
}
