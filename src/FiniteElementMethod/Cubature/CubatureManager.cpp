#include "FiniteElementMethod/Cubature/CubatureManager.hpp"
#include "FiniteElementMethod/Cell/CellSelector.hpp"
#include "FiniteElementMethod/Cubature/BasisSelector.hpp"
#include "FiniteElementMethod/FEMExceptions.hpp"
#include "FiniteElementMethod/FEMTypes.hpp"
#include "FiniteElementMethod/Utils/FEMCheckers.hpp"
#include "FiniteElementMethod/Utils/FEMLimits.hpp"
#include "Utilities/Utilities.hpp"

void CubatureManager::_initializeCubature(CellType cell_type, short desired_accuracy, short polynom_order)
{
    try
    {
        // 1. Defining cell topology as tetrahedron.
        auto cellTopology{CellSelector::get(cell_type)};
        auto basis{BasisSelector::template get<DeviceType>(cell_type, polynom_order)};
        m_count_basis_functions = basis->getCardinality(); // For linear tetrahedrons (polynom order = 1) count of basis functions = 4 (4 verteces, 4 basis functions).

        // 2. Using cubature factory to create cubature function.
        Intrepid2::DefaultCubatureFactory cubFactory;
        auto cubature{cubFactory.create<DeviceType>(cellTopology, desired_accuracy)}; // Generating cubature function.
        m_count_cubature_points = cubature->getNumPoints();                           // Getting number of cubature points.

        // 3. Allocating memory for cubature points and weights.
        m_cubature_points = DynRankView("cubPoints", m_count_cubature_points, FEM_LIMITS_DEFAULT_SPACE_DIMENSION); // Matrix: m_count_cubature_points x Dimensions.
        m_cubature_weights = DynRankView("cubWeights", m_count_cubature_points);                                   // Vector: m_count_cubature_points.

        // 4. Getting cubature points and weights.
        cubature->getCubature(m_cubature_points, m_cubature_weights);
    }
    catch (std::exception const &ex)
    {
        std::string errorMessage{util::stringify("Error while trying to initialize cubature. ",
                                                 "Please check input parameters: ",
                                                 "\ncell_type = ",
                                                 static_cast<short>(cell_type),
                                                 "\ndesired_accuracy = ",
                                                 desired_accuracy,
                                                 "\npolynom_order = ",
                                                 polynom_order,
                                                 "\n\n",
                                                 ex.what())};
        ERRMSG(errorMessage);
        START_THROW_EXCEPTION(CubatureInitializingCubatureException, errorMessage);
    }
    catch (...)
    {
        std::string errorMessage{util::stringify("Unknown error while trying to initialize cubature. ",
                                                 "Please check input parameters: ",
                                                 "\ncell_type = ",
                                                 static_cast<short>(cell_type),
                                                 "\ndesired_accuracy = ",
                                                 desired_accuracy,
                                                 "\npolynom_order = ",
                                                 polynom_order,
                                                 "\n\n")};
        ERRMSG(errorMessage);
        START_THROW_EXCEPTION(CubatureUnknownException, errorMessage);
    }
}

CubatureManager::CubatureManager(CellType cell_type, short desired_accuracy, short polynom_order)
{
    FEMCheckers::checkCellType(cell_type);
    FEMCheckers::checkDesiredAccuracy(desired_accuracy);
    FEMCheckers::checkPolynomOrder(polynom_order);

    _initializeCubature(cell_type, desired_accuracy, polynom_order);
}
