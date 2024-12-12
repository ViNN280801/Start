#include "FiniteElementMethod/GSMAssembler.hpp"
#include "FiniteElementMethod/BoundaryConditions/BoundaryConditionsManager.hpp"
#include "FiniteElementMethod/Cell/CellSelector.hpp"
#include "FiniteElementMethod/Cubature/BasisSelector.hpp"
#include "FiniteElementMethod/FEMCheckers.hpp"
#include "FiniteElementMethod/FEMLimits.hpp"

std::vector<MatrixEntry> GSMAssembler::_getMatrixEntries()
{
    TetrahedronIndicesVector globalNodeIndicesPerElement;
    std::set<size_t> allNodeIDs;
    for (auto const &tetrahedronData : m_meshManager.getMeshComponents())
        for (auto const &nodeData : tetrahedronData.nodes)
            allNodeIDs.insert(nodeData.globalNodeId);
    for (auto const &tetrahedronData : m_meshManager.getMeshComponents())
    {
        std::array<LocalOrdinal, 4ul> nodes;
        for (short i{}; i < FEM_LIMITS_DEFAULT_TETRAHEDRON_VERTICES_COUNT; ++i)
            nodes[i] = tetrahedronData.nodes[i].globalNodeId - 1;
        globalNodeIndicesPerElement.emplace_back(nodes);
    }

// 1. Getting all LSMs.
#ifdef USE_CUDA
    auto localStiffnessMatrices_device{_computeLocalStiffnessMatrices()};
    auto localStiffnessMatrices{Kokkos::create_mirror_view_and_copy(MemorySpaceHost(), localStiffnessMatrices_device)}; // Transfering LSMs from device to host.
#else
    auto localStiffnessMatrices{_computeLocalStiffnessMatrices()};
#endif

    // 2. Filling matrix entries.
    std::vector<MatrixEntry> matrixEntries;
    try
    {
        for (size_t tetraId{}; tetraId < globalNodeIndicesPerElement.size(); ++tetraId)
        {
            auto const &nodeIndices{globalNodeIndicesPerElement.at(tetraId)};
            for (size_t localNodeI{}; localNodeI < FEM_LIMITS_DEFAULT_TETRAHEDRON_VERTICES_COUNT; ++localNodeI)
            {
                for (size_t localNodeJ{}; localNodeJ < FEM_LIMITS_DEFAULT_TETRAHEDRON_VERTICES_COUNT; ++localNodeJ)
                {
                    Scalar value{localStiffnessMatrices(tetraId, localNodeI, localNodeJ)};
                    GlobalOrdinal globalRow{nodeIndices[localNodeI]},
                        globalCol{nodeIndices[localNodeJ]};

                    matrixEntries.push_back({globalRow, globalCol, value});
                }
            }
        }
    }
    catch (std::exception const &ex)
    {
        ERRMSG(ex.what());
    }
    catch (...)
    {
        ERRMSG("Unknown error was occured");
    }

    if (matrixEntries.empty())
    {
        WARNINGMSG("Something went wrong while filling matrix entries - matrix entries are empty - there is no elements");
    }

    return matrixEntries;
}

DynRankView GSMAssembler::_computeLocalStiffnessMatrices()
{
    try
    {
        auto const numBasisFunctions{m_cubature_manager.getCountBasisFunctions()};
        auto const numCubaturePoints{m_cubature_manager.getCountCubaturePoints()};

        auto const &cubPoints{m_cubature_manager.getCubaturePoints()};
        auto const &cubWeights{m_cubature_manager.getCubatureWeights()};

        auto basis{BasisSelector::template get<DeviceType>(CellType::Tetrahedron, m_polynom_order)};
        return m_meshManager.computeLocalStiffnessMatricesAndNablaPhi(
            basis.get(),
            cubPoints,
            cubWeights,
            numBasisFunctions,
            numCubaturePoints);
    }
    catch (std::exception const &ex)
    {
        ERRMSG(ex.what());
    }
    catch (...)
    {
        ERRMSG("Unknown error");
    }
    WARNINGMSG("Returning empty multidimensional array which was intended for LSM (Local Stiffness Matrix)");
    return DynRankView();
}

GSMAssembler::GSMAssembler(std::string_view mesh_filename, CellType cell_type, short desired_calc_accuracy, short polynom_order)
    : m_mesh_filename(mesh_filename.data()),
      m_cell_type(cell_type),
      m_desired_accuracy(desired_calc_accuracy),
      m_polynom_order(polynom_order),
      m_meshManager(mesh_filename),
      m_cubature_manager(cell_type, desired_calc_accuracy, polynom_order),
      m_matrix_manager(_getMatrixEntries())
{
    FEMCheckers::checkMeshFile(mesh_filename);
    FEMCheckers::checkPolynomOrder(polynom_order);
    FEMCheckers::checkDesiredAccuracy(desired_calc_accuracy);
}
