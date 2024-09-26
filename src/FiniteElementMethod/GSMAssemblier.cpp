#include "FiniteElementMethod/GSMAssemblier.hpp"
#include "FiniteElementMethod/BoundaryConditions/BoundaryConditionsManager.hpp"
#include "FiniteElementMethod/Cell/CellSelector.hpp"
#include "FiniteElementMethod/Cubature/BasisSelector.hpp"
#include "FiniteElementMethod/FEMCheckers.hpp"
#include "FiniteElementMethod/FEMLimits.hpp"
#include "Generators/RealNumberGenerator.hpp"

DynRankView GSMAssemblier::_getTetrahedronVertices()
{
    try
    {
        DynRankView vertices("vertices", getMeshManager().getNumTetrahedrons(), FEM_LIMITS_DEFAULT_TETRAHEDRON_VERTICES_COUNT, FEM_LIMITS_DEFAULT_SPACE_DIMENSION);
        Kokkos::deep_copy(vertices, 0.0);

        size_t i{};
        for (auto const &meshParam : getMeshManager().getMeshComponents())
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
    return DynRankView();
}

std::vector<MatrixEntry> GSMAssemblier::_getMatrixEntries()
{
    TetrahedronIndicesVector globalNodeIndicesPerElement;
    std::set<size_t> allNodeIDs;
    for (auto const &tetrahedronData : getMeshManager().getMeshComponents())
        for (auto const &nodeData : tetrahedronData.nodes)
            allNodeIDs.insert(nodeData.globalNodeId);
    for (auto const &tetrahedronData : getMeshManager().getMeshComponents())
    {
        std::array<LocalOrdinal, 4ul> nodes;
        for (short i{}; i < FEM_LIMITS_DEFAULT_TETRAHEDRON_VERTICES_COUNT; ++i)
            nodes[i] = tetrahedronData.nodes[i].globalNodeId - 1;
        globalNodeIndicesPerElement.emplace_back(nodes);
    }

    // 1. Getting all LSMs.
    auto localStiffnessMatrices{_computeLocalStiffnessMatrices()};

    // 3. Filling matrix entries.
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
        WARNINGMSG("Something went wrong while filling matrix entries - matrix entries are empty - there is no elements");

    return matrixEntries;
}

DynRankView GSMAssemblier::_computeLocalStiffnessMatrices()
{
    try
    {
        // 1. Calculating basis gradients.
        DynRankView referenceBasisGrads("referenceBasisGrads", m_cubature_manager.getCountBasisFunctions(), m_cubature_manager.getCountCubaturePoints(), FEM_LIMITS_DEFAULT_SPACE_DIMENSION);
        Kokkos::deep_copy(referenceBasisGrads, 0.0);
        auto basis{BasisSelector::template get<DeviceType>(CellType::Tetrahedron, m_polynom_order)};
        basis->getValues(referenceBasisGrads, m_cubature_manager.getCubaturePoints(), Intrepid2::OPERATOR_GRAD);

        // 2. Computing cell jacobians, inversed jacobians and jacobian determinants to get cell measure.
        DynRankView jacobians("jacobians", getMeshManager().getNumTetrahedrons(), m_cubature_manager.getCountCubaturePoints(), FEM_LIMITS_DEFAULT_SPACE_DIMENSION, FEM_LIMITS_DEFAULT_SPACE_DIMENSION);
        Kokkos::deep_copy(jacobians, 0.0);
        Intrepid2::CellTools<DeviceType>::setJacobian(jacobians, m_cubature_manager.getCubaturePoints(), _getTetrahedronVertices(), CellSelector::get(CellType::Tetrahedron));

        DynRankView invJacobians("invJacobians", getMeshManager().getNumTetrahedrons(), m_cubature_manager.getCountCubaturePoints(), FEM_LIMITS_DEFAULT_SPACE_DIMENSION, FEM_LIMITS_DEFAULT_SPACE_DIMENSION);
        Kokkos::deep_copy(invJacobians, 0.0);
        Intrepid2::CellTools<DeviceType>::setJacobianInv(invJacobians, jacobians);

        DynRankView jacobiansDet("jacobiansDet", getMeshManager().getNumTetrahedrons(), m_cubature_manager.getCountCubaturePoints());
        Kokkos::deep_copy(jacobiansDet, 0.0);
        Intrepid2::CellTools<DeviceType>::setJacobianDet(jacobiansDet, jacobians);
        DynRankView cellMeasures("cellMeasures", getMeshManager().getNumTetrahedrons(), m_cubature_manager.getCountCubaturePoints());
        Kokkos::deep_copy(cellMeasures, 0.0);
        Intrepid2::FunctionSpaceTools<DeviceType>::computeCellMeasure(cellMeasures, jacobiansDet, m_cubature_manager.getCubatureWeights());

        // 3. Transforming reference basis gradients to physical frame.
        DynRankView transformedBasisGradients("transformedBasisGradients", getMeshManager().getNumTetrahedrons(), m_cubature_manager.getCountBasisFunctions(), m_cubature_manager.getCountCubaturePoints(), FEM_LIMITS_DEFAULT_SPACE_DIMENSION);
        Kokkos::deep_copy(transformedBasisGradients, 0.0);
        Intrepid2::FunctionSpaceTools<DeviceType>::HGRADtransformGRAD(transformedBasisGradients, invJacobians, referenceBasisGrads);

        // 4. Multiply transformed basis gradients by cell measures to get weighted gradients.
        DynRankView weightedBasisGrads("weightedBasisGrads", getMeshManager().getNumTetrahedrons(), m_cubature_manager.getCountBasisFunctions(), m_cubature_manager.getCountCubaturePoints(), FEM_LIMITS_DEFAULT_SPACE_DIMENSION);
        Kokkos::deep_copy(weightedBasisGrads, 0.0);
        Intrepid2::FunctionSpaceTools<DeviceType>::multiplyMeasure(weightedBasisGrads, cellMeasures, transformedBasisGradients);

        // 5. Integrate to get local stiffness matrices for workset cells.
        DynRankView localStiffnessMatrices("localStiffnessMatrices", getMeshManager().getNumTetrahedrons(), m_cubature_manager.getCountBasisFunctions(), m_cubature_manager.getCountBasisFunctions());
        Kokkos::deep_copy(localStiffnessMatrices, 0.0);
        Intrepid2::FunctionSpaceTools<DeviceType>::integrate(localStiffnessMatrices, weightedBasisGrads, transformedBasisGradients);

        // Filling map with basis grads on each node on each tetrahedron.
        for (size_t localTetraId{}; localTetraId < getMeshManager().getNumTetrahedrons(); ++localTetraId)
        {
            auto globalTetraId{getMeshManager().getGlobalTetraId(localTetraId)};

            auto const &nodes {getMeshManager().getTetrahedronNodes(localTetraId)};
            if (m_cubature_manager.getCountBasisFunctions() > nodes.size())
            {
                ERRMSG("Basis function count exceeds the number of nodes.");
            }

            for (short localNodeId{}; localNodeId < m_cubature_manager.getCountBasisFunctions(); ++localNodeId)
            {
                auto globalNodeId{getMeshManager().getGlobalNodeId(localTetraId, localNodeId)};
                Point grad(weightedBasisGrads(localTetraId, localNodeId, 0, 0),
                           weightedBasisGrads(localTetraId, localNodeId, 0, 1),
                           weightedBasisGrads(localTetraId, localNodeId, 0, 2));

                // As we have polynom order = 1, that all the values from the ∇φ in all cub points are the same, so we can add only 1 row from each ∇φ.
                getMeshManager().assignNablaPhi(globalTetraId, globalNodeId, grad);
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
      m_cubature_manager(cell_type, desired_calc_accuracy, polynom_order),
      m_matrix_manager(_getMatrixEntries())
{
    FEMCheckers::checkMeshFile(mesh_filename);
    FEMCheckers::checkPolynomOrder(polynom_order);
    FEMCheckers::checkDesiredAccuracy(desired_calc_accuracy);

    _assembleGlobalStiffnessMatrix();
}
