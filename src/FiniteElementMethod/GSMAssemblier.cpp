#include "FiniteElementMethod/GSMAssemblier.hpp"
#include "FiniteElementMethod/BoundaryConditions/BoundaryConditionsManager.hpp"
#include "FiniteElementMethod/Cell/CellSelector.hpp"
#include "FiniteElementMethod/Cubature/BasisSelector.hpp"
#include "FiniteElementMethod/FEMCheckers.hpp"
#include "FiniteElementMethod/FEMLimits.hpp"
#include "Generators/RealNumberGenerator.hpp"

DynRankViewHost GSMAssemblier::_getTetrahedronVertices()
{
    try
    {
        DynRankViewHost vertices("vertices", m_meshManager.getNumTetrahedrons(), FEM_LIMITS_DEFAULT_TETRAHEDRON_VERTICES_COUNT, FEM_LIMITS_DEFAULT_SPACE_DIMENSION);
        size_t i{};
        for (auto const &meshParam : m_meshManager.getMeshComponents())
        {
            auto tetrahedron{meshParam.tetrahedron};
            for (short node{}; node < FEM_LIMITS_DEFAULT_TETRAHEDRON_VERTICES_COUNT; ++node)
            {
                vertices(i, node, 0) = CGAL_TO_DOUBLE(tetrahedron.vertex(node).x());
                vertices(i, node, 1) = CGAL_TO_DOUBLE(tetrahedron.vertex(node).y());
                vertices(i, node, 2) = CGAL_TO_DOUBLE(tetrahedron.vertex(node).z());
            }
            ++i;
        }
        return vertices;
    }
    catch (std::exception const &ex)
    {
        ERRMSG(ex.what());
    }
    catch (...)
    {
        ERRMSG("Unknown error");
    }
    WARNINGMSG("Returning empty multidimensional array with vertices of the all tetrahedrons from the mesh");
    return DynRankViewHost();
}

std::vector<MatrixEntry> GSMAssemblier::_getMatrixEntries()
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

DynRankView GSMAssemblier::_computeLocalStiffnessMatrices()
{
    try
    {
        auto const numTetrahedrons{m_meshManager.getNumTetrahedrons()};
        auto const numBasisFunctions{m_cubature_manager.getCountBasisFunctions()};
        auto const numCubaturePoints{m_cubature_manager.getCountCubaturePoints()};

        auto const &cubPoints{m_cubature_manager.getCubaturePoints()};
        auto const &cubWeights{m_cubature_manager.getCubatureWeights()};

        // 1. Calculating basis gradients.
        DynRankView referenceBasisGrads("referenceBasisGrads", numBasisFunctions, numCubaturePoints, FEM_LIMITS_DEFAULT_SPACE_DIMENSION);
        auto basis{BasisSelector::template get<DeviceType>(CellType::Tetrahedron, m_polynom_order)};
        basis->getValues(referenceBasisGrads, cubPoints, Intrepid2::OPERATOR_GRAD);

#ifdef USE_CUDA
        auto vertices{Kokkos::create_mirror_view_and_copy(MemorySpaceDevice(), _getTetrahedronVertices())}; // From host to device.
#else
        DynRankView vertices(_getTetrahedronVertices());
#endif

        // 2. Computing cell jacobians, inversed jacobians and jacobian determinants to get cell measure.
        DynRankView jacobians("jacobians", numTetrahedrons, numCubaturePoints, FEM_LIMITS_DEFAULT_SPACE_DIMENSION, FEM_LIMITS_DEFAULT_SPACE_DIMENSION);
        Intrepid2::CellTools<DeviceType>::setJacobian(jacobians, cubPoints, vertices, CellSelector::get(CellType::Tetrahedron));

        DynRankView invJacobians("invJacobians", numTetrahedrons, numCubaturePoints, FEM_LIMITS_DEFAULT_SPACE_DIMENSION, FEM_LIMITS_DEFAULT_SPACE_DIMENSION);
        Intrepid2::CellTools<DeviceType>::setJacobianInv(invJacobians, jacobians);

        DynRankView jacobiansDet("jacobiansDet", numTetrahedrons, numCubaturePoints);
        Intrepid2::CellTools<DeviceType>::setJacobianDet(jacobiansDet, jacobians);

        DynRankView cellMeasures("cellMeasures", numTetrahedrons, numCubaturePoints);
        Intrepid2::FunctionSpaceTools<DeviceType>::computeCellMeasure(cellMeasures, jacobiansDet, cubWeights);

        // 3. Transforming reference basis gradients to physical frame.
        DynRankView transformedBasisGradients("transformedBasisGradients", numTetrahedrons, numBasisFunctions, numCubaturePoints, FEM_LIMITS_DEFAULT_SPACE_DIMENSION);
        Intrepid2::FunctionSpaceTools<DeviceType>::HGRADtransformGRAD(transformedBasisGradients, invJacobians, referenceBasisGrads);

        // 4. Multiply transformed basis gradients by cell measures to get weighted gradients.
        DynRankView weightedBasisGrads("weightedBasisGrads", numTetrahedrons, numBasisFunctions, numCubaturePoints, FEM_LIMITS_DEFAULT_SPACE_DIMENSION);
        Intrepid2::FunctionSpaceTools<DeviceType>::multiplyMeasure(weightedBasisGrads, cellMeasures, transformedBasisGradients);

        // 5. Integrate to get local stiffness matrices for workset cells.
        DynRankView localStiffnessMatrices("localStiffnessMatrices", numTetrahedrons, numBasisFunctions, numBasisFunctions);
        Intrepid2::FunctionSpaceTools<DeviceType>::integrate(localStiffnessMatrices, weightedBasisGrads, transformedBasisGradients);

#ifdef USE_CUDA
        auto localStiffnessMatrices_host{Kokkos::create_mirror_view_and_copy(MemorySpaceHost(), localStiffnessMatrices)}; // From device to host.
        auto weightedBasisGrads_host{Kokkos::create_mirror_view_and_copy(MemorySpaceHost(), weightedBasisGrads)};         // From device to host.
#endif

        // 6. Filling map with basis grads on each node on each tetrahedron.
        for (size_t localTetraId{}; localTetraId < numTetrahedrons; ++localTetraId)
        {
            auto globalTetraId{m_meshManager.getGlobalTetraId(localTetraId)};
            auto const &nodes{m_meshManager.getTetrahedronNodes(localTetraId)};
            if (numBasisFunctions > nodes.size())
            {
                ERRMSG("Basis function count exceeds the number of nodes.");
            }

            for (short localNodeId{}; localNodeId < numBasisFunctions; ++localNodeId)
            {
                auto globalNodeId{m_meshManager.getGlobalNodeId(localTetraId, localNodeId)};

#ifdef USE_CUDA
                Point grad(weightedBasisGrads_host(localTetraId, localNodeId, 0, 0),
                           weightedBasisGrads_host(localTetraId, localNodeId, 0, 1),
                           weightedBasisGrads_host(localTetraId, localNodeId, 0, 2));
#else
                Point grad(weightedBasisGrads(localTetraId, localNodeId, 0, 0),
                           weightedBasisGrads(localTetraId, localNodeId, 0, 1),
                           weightedBasisGrads(localTetraId, localNodeId, 0, 2));
#endif

                /// As we have polynomial order \( = 1 \), all the values from \( \nabla \varphi \)
                /// in all cubature points are the same, so we can add only one row from each \( \nabla \varphi \).
                m_meshManager.assignNablaPhi(globalTetraId, globalNodeId, grad);
            }
        }
        return localStiffnessMatrices;
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

void GSMAssemblier::_assembleGlobalStiffnessMatrix()
{
    try
    {
        // 1. Preparing to fill matrix.
        getGlobalStiffnessMatrix()->resumeFill();

        // 2. Adding local stiffness matrices to the global.
        for (auto const &entry : m_matrix_manager.getMatrixEntries())
        {
            Teuchos::ArrayView<GlobalOrdinal const> colsView(std::addressof(entry.col), 1);
            Teuchos::ArrayView<Scalar const> valsView(std::addressof(entry.value), 1);
            getGlobalStiffnessMatrix()->sumIntoGlobalValues(entry.row, colsView, valsView);
        }

        // 3. Filling completion.
        getGlobalStiffnessMatrix()->fillComplete();
    }
    catch (std::exception const &ex)
    {
        ERRMSG(ex.what());
    }
    catch (...)
    {
        ERRMSG("Unknown error was occured while assemblying global stiffness matrix. Probably solution: decrease polynom order or desired accuracy");
    }
}

GSMAssemblier::GSMAssemblier(std::string_view mesh_filename, CellType cell_type, short desired_calc_accuracy, short polynom_order)
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

    _assembleGlobalStiffnessMatrix();
}
