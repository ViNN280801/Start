#include <gmsh.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "DataHandling/TetrahedronMeshManager.hpp"
#include "FiniteElementMethod/Cell/CellSelector.hpp"
#include "FiniteElementMethod/FEMLimits.hpp"
#include "Geometry/MathVector.hpp"

Point TetrahedronMeshManager::TetrahedronData::getTetrahedronCenter() const
{
    try
    {
        double x{}, y{}, z{};
        for (auto const &node : nodes)
        {
            x += node.nodeCoords.x();
            y += node.nodeCoords.y();
            z += node.nodeCoords.z();
        }
        return Point{x / 4.0, y / 4.0, z / 4.0};
    }
    catch (std::exception const &e)
    {
        throw std::runtime_error(util::stringify("Error computing tetrahedron center: ", e.what()));
    }
}

void TetrahedronMeshManager::_readAndPartitionMesh(std::string_view mesh_filename)
{
    // Only called by rank 0.
    try
    {
        gmsh::open(mesh_filename.data());

        std::vector<std::size_t> nodeTags;
        std::vector<double> coord, parametricCoord;
        gmsh::model::mesh::getNodes(nodeTags, coord, parametricCoord);

        m_nodeCoordinatesMap.clear();
        for (size_t i{}; i < nodeTags.size(); ++i)
        {
            size_t nodeID{nodeTags[i]};
            std::array<double, 3> coords = {coord[i * 3 + 0], coord[i * 3 + 1], coord[i * 3 + 2]};
            m_nodeCoordinatesMap[nodeID] = coords;
        }

        // Read tetrahedron elements (element type 4 corresponds to tetrahedra).
        std::vector<size_t> elTags, nodeTagsByEl;
        gmsh::model::mesh::getElementsByType(4, elTags, nodeTagsByEl);

        // Partition elements among MPI processes.
        size_t numElements{elTags.size()};
        m_elementsPerProc.resize(m_size);

        for (size_t i{}; i < numElements; ++i)
        {
            int destRank = i % m_size;
            size_t elID{elTags[i]};
            std::array<size_t, 4> nodeIDs;
            for (short j{}; j < 4; ++j)
                nodeIDs[j] = nodeTagsByEl[i * 4 + j];
            m_elementsPerProc[destRank].push_back(std::make_pair(elID, nodeIDs));
        }

        // Collect node IDs per process.
        m_nodeTagsPerProc.resize(m_size);

        for (int proc{}; proc < m_size; ++proc)
        {
            std::set<size_t> nodeSet;
            for (auto const &el : m_elementsPerProc[proc])
            {
                auto const &nodeIDs{el.second};
                nodeSet.insert(nodeIDs.begin(), nodeIDs.end());
            }
            m_nodeTagsPerProc[proc].assign(nodeSet.begin(), nodeSet.end());
        }

        // Store local data for rank 0.
        m_localElementData = m_elementsPerProc[0]; // Vector of pair<size_t, array<size_t,4>>.
        m_localNodeTags = m_nodeTagsPerProc[0];

        // Collect node coordinates for local nodes.
        for (size_t nodeID : m_localNodeTags)
            m_localNodeCoordinates[nodeID] = m_nodeCoordinatesMap[nodeID];
    }
    catch (std::exception const &e)
    {
        throw std::runtime_error(std::string("Error in _readAndPartitionMesh: ") + e.what());
    }
}

void TetrahedronMeshManager::_distributeMeshData()
{
#ifdef USE_MPI
    // Only called by rank 0.
    try
    {
        for (int destRank{1}; destRank < m_size; ++destRank)
        {
            // Send the number of elements.
            size_t numLocalElements{m_elementsPerProc[destRank].size()};
            MPI_Send(std::addressof(numLocalElements), 1, MPI_UNSIGNED_LONG, destRank, 0, MPI_COMM_WORLD);

            // Prepare element data to send: for each element, element ID and node IDs.
            // Total data size: numLocalElements * (1 + 4) size_t.
            std::vector<size_t> elementData; // Flat array of element IDs and node IDs.
            elementData.reserve(numLocalElements * 5);
            for (auto const &el : m_elementsPerProc[destRank])
            {
                elementData.push_back(el.first);                                           // Element ID.
                elementData.insert(elementData.end(), el.second.begin(), el.second.end()); // Node IDs.
            }

            // Send element data.
            MPI_Send(elementData.data(), elementData.size(), MPI_UNSIGNED_LONG, destRank, 1, MPI_COMM_WORLD);

            // Send node IDs.
            size_t numLocalNodes{m_nodeTagsPerProc[destRank].size()};
            MPI_Send(std::addressof(numLocalNodes), 1, MPI_UNSIGNED_LONG, destRank, 2, MPI_COMM_WORLD);
            MPI_Send(m_nodeTagsPerProc[destRank].data(), numLocalNodes, MPI_UNSIGNED_LONG, destRank, 3, MPI_COMM_WORLD);

            // Collect coordinates for the nodes used.
            std::vector<double> localNodeCoords;
            for (size_t nodeID : m_nodeTagsPerProc[destRank])
            {
                auto coords{m_nodeCoordinatesMap[nodeID]};
                localNodeCoords.insert(localNodeCoords.end(), coords.begin(), coords.end());
            }

            // Send node coordinates.
            MPI_Send(localNodeCoords.data(), localNodeCoords.size(), MPI_DOUBLE, destRank, 4, MPI_COMM_WORLD);
        }
    }
    catch (std::exception const &e)
    {
        throw std::runtime_error(std::string("Error in _distributeMeshData: ") + e.what());
    }
#endif
}

void TetrahedronMeshManager::_receiveData()
{
#ifdef USE_MPI
    // Called by ranks != 0.
    try
    {
        // Receive the number of elements.
        size_t numLocalElements;
        MPI_Recv(std::addressof(numLocalElements), 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Receive element data.
        std::vector<size_t> elementData(numLocalElements * 5);
        MPI_Recv(elementData.data(), elementData.size(), MPI_UNSIGNED_LONG, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Reconstruct m_localElementData.
        m_localElementData.resize(numLocalElements);
        for (size_t i{}; i < numLocalElements; ++i)
        {
            size_t elID = elementData[i * 5];
            std::array<size_t, 4> nodeIDs;
            for (short j{}; j < 4; ++j)
                nodeIDs[j] = elementData[i * 5 + 1 + j];
            m_localElementData[i] = std::make_pair(elID, nodeIDs);
        }

        // Receive node IDs.
        size_t numLocalNodes;
        MPI_Recv(std::addressof(numLocalNodes), 1, MPI_UNSIGNED_LONG, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        m_localNodeTags.resize(numLocalNodes);
        MPI_Recv(m_localNodeTags.data(), numLocalNodes, MPI_UNSIGNED_LONG, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Receive node coordinates.
        std::vector<double> localNodeCoords(numLocalNodes * 3);
        MPI_Recv(localNodeCoords.data(), localNodeCoords.size(), MPI_DOUBLE, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Reconstruct m_localNodeCoordinates.
        for (size_t i{}; i < numLocalNodes; ++i)
        {
            size_t nodeID{m_localNodeTags[i]};
            std::array<double, 3> coords = {
                localNodeCoords[i * 3 + 0],
                localNodeCoords[i * 3 + 1],
                localNodeCoords[i * 3 + 2]};
            m_localNodeCoordinates[nodeID] = coords;
        }
    }
    catch (std::exception const &e)
    {
        throw std::runtime_error(std::string("Error in _receiveData: ") + e.what());
    }
#endif
}

void TetrahedronMeshManager::_constructLocalMesh()
{
    // Use m_localElementData and m_localNodeCoordinates to construct m_meshComponents.
    try
    {
        for (auto const &elData : m_localElementData)
        {
            size_t tetrahedronID{elData.first};
            std::array<size_t, 4> const &nodeIDs = elData.second;

            std::array<Point, 4> vertices;
            for (short j{}; j < 4; ++j)
            {
                size_t nodeID = nodeIDs[j];
                auto coords = m_localNodeCoordinates.at(nodeID);
                vertices[j] = Point(coords[0], coords[1], coords[2]);
            }

            Tetrahedron tetrahedron(vertices[0], vertices[1], vertices[2], vertices[3]);
            TetrahedronData data{tetrahedronID, tetrahedron, {}, std::nullopt};

            for (short j{}; j < 4; ++j)
                data.nodes[j] = {nodeIDs[j], vertices[j], std::nullopt, std::nullopt};

            m_meshComponents.emplace_back(data);
        }
    }
    catch (std::exception const &e)
    {
        throw std::runtime_error(std::string("Error in _constructLocalMesh: ") + e.what());
    }
}

void TetrahedronMeshManager::_buildViews()
{
    m_tetraVertices = DynRankViewHost("tetraVertices", getNumTetrahedrons(), 4, 3);
    m_tetraGlobalIds = DynRankViewHost("tetraGlobalIds", getNumTetrahedrons());

    for (size_t localId{}; localId < getNumTetrahedrons(); ++localId)
    {
        auto const &data{m_meshComponents[localId]};
        m_tetraGlobalIds(localId) = data.globalTetraId;
        for (short node{}; node < 4; ++node)
        {
            m_tetraVertices(localId, node, 0) = data.nodes[node].nodeCoords.x();
            m_tetraVertices(localId, node, 1) = data.nodes[node].nodeCoords.y();
            m_tetraVertices(localId, node, 2) = data.nodes[node].nodeCoords.z();
        }
    }

    m_globalToLocalTetraId.reserve(getNumTetrahedrons());
    for (size_t localId{}; localId < getNumTetrahedrons(); ++localId)
        m_globalToLocalTetraId[m_tetraGlobalIds(localId)] = localId;
}

TetrahedronMeshManager::TetrahedronMeshManager(std::string_view mesh_filename)
{
    util::check_gmsh_mesh_file(mesh_filename);

    m_rank = 0;
    m_size = 1;
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, std::addressof(m_rank));
    MPI_Comm_size(MPI_COMM_WORLD, std::addressof(m_size));
#endif

    if (m_rank == 0)
    {
        _readAndPartitionMesh(mesh_filename);
        _distributeMeshData();
    }
    else
        _receiveData();

    _constructLocalMesh();
    _buildViews();
}

void TetrahedronMeshManager::print() const noexcept
{
    for (auto const &meshComponent : m_meshComponents)
    {
        std::cout << "Tetrahedron[" << meshComponent.globalTetraId << "]\n";

        for (short i{}; i < 4; ++i)
        {
            std::cout << "Vertex[" << meshComponent.nodes.at(i).globalNodeId << "]: ("
                      << meshComponent.nodes.at(i).nodeCoords.x() << ", "
                      << meshComponent.nodes.at(i).nodeCoords.y() << ", "
                      << meshComponent.nodes.at(i).nodeCoords.z() << ")\n";

            if (meshComponent.nodes.at(i).nablaPhi)
            {
                std::cout << "  ∇φ: ("
                          << meshComponent.nodes.at(i).nablaPhi->x() << ", "
                          << meshComponent.nodes.at(i).nablaPhi->y() << ", "
                          << meshComponent.nodes.at(i).nablaPhi->z() << ")\n";
            }
            else
            {
                std::cout << "  ∇φ: empty\n";
            }

            if (meshComponent.nodes.at(i).potential)
            {
                std::cout << "  Potential φ: " << meshComponent.nodes.at(i).potential.value() << "\n";
            }
            else
            {
                std::cout << "  Potential φ: empty\n";
            }
        }

        if (meshComponent.electricField)
        {
            std::cout << "ElectricField: ("
                      << meshComponent.electricField->x() << ", "
                      << meshComponent.electricField->y() << ", "
                      << meshComponent.electricField->z() << ")\n";
        }
        else
        {
            std::cout << "ElectricField: empty\n";
        }
    }

    std::cout << std::endl;
}

std::optional<TetrahedronMeshManager::TetrahedronData> TetrahedronMeshManager::getMeshDataByTetrahedronId(size_t globalTetrahedronId) const
{
#if __cplusplus >= 202002L
    auto it{std::ranges::find_if(m_meshComponents, [globalTetrahedronId](const TetrahedronData &data)
                                 { return data.globalTetraId == globalTetrahedronId; })};
#else
    auto it{std::find_if(m_meshComponents.cbegin(), m_meshComponents.cend(), [globalTetrahedronId](const TetrahedronData &data)
                         { return data.globalTetraId == globalTetrahedronId; })};
#endif
    if (it != m_meshComponents.cend())
        return *it;
    return std::nullopt;
}

void TetrahedronMeshManager::assignNablaPhi(size_t tetrahedronId, size_t nodeId, Point const &gradient)
{
#if __cplusplus >= 202002L
    auto it{std::ranges::find_if(m_meshComponents, [tetrahedronId](const TetrahedronData &data)
                                 { return data.globalTetraId == tetrahedronId; })};
#else
    auto it{std::find_if(m_meshComponents.begin(), m_meshComponents.end(), [tetrahedronId](const TetrahedronData &data)
                         { return data.globalTetraId == tetrahedronId; })};
#endif
    if (it != m_meshComponents.cend())
    {
        for (auto &node : it->nodes)
        {
            if (node.globalNodeId == nodeId)
            {
                node.nablaPhi = gradient;
                return;
            }
        }
    }
}

void TetrahedronMeshManager::assignPotential(size_t nodeId, double potential)
{
    for (auto &tetrahedron : m_meshComponents)
        for (auto &node : tetrahedron.nodes)
            if (node.globalNodeId == nodeId)
                node.potential = potential;
}

void TetrahedronMeshManager::assignElectricField(size_t tetrahedronId, Point const &electricField)
{
#if __cplusplus >= 202002L
    auto it{std::ranges::find_if(m_meshComponents, [tetrahedronId](const TetrahedronData &data)
                                 { return data.globalTetraId == tetrahedronId; })};
#else
    auto it{std::find_if(m_meshComponents.begin(), m_meshComponents.end(), [tetrahedronId](const TetrahedronData &data)
                         { return data.globalTetraId == tetrahedronId; })};
#endif
    if (it != m_meshComponents.end())
    {
        it->electricField = electricField;
    }
}

std::map<size_t, std::vector<size_t>> TetrahedronMeshManager::getTetrahedronNodesMap() const
{
    std::map<size_t, std::vector<size_t>> tetrahedronNodesMap;

    for (auto const &meshData : m_meshComponents)
        for (short i{}; i < 4; ++i)
            tetrahedronNodesMap[meshData.globalTetraId].emplace_back(meshData.nodes.at(i).globalNodeId);

    if (tetrahedronNodesMap.empty())
    {
        WARNINGMSG("Tetrahedron - nodes map is empty");
    }
    return tetrahedronNodesMap;
}

std::map<size_t, std::vector<size_t>> TetrahedronMeshManager::getNodeTetrahedronsMap() const
{
    std::map<size_t, std::vector<size_t>> nodeTetrahedronsMap;

    for (auto const &meshData : m_meshComponents)
        for (short i{}; i < 4; ++i)
            nodeTetrahedronsMap[meshData.nodes.at(i).globalNodeId].emplace_back(meshData.globalTetraId);

    if (nodeTetrahedronsMap.empty())
    {
        WARNINGMSG("Node - tetrahedrons map is empty");
    }
    return nodeTetrahedronsMap;
}

std::map<size_t, Point> TetrahedronMeshManager::getTetrahedronCenters() const
{
    std::map<size_t, Point> tetraCentres;

    for (auto const &meshData : m_meshComponents)
        tetraCentres[meshData.globalTetraId] = meshData.getTetrahedronCenter();

    if (tetraCentres.empty())
    {
        WARNINGMSG("Tetrahedron centres map is empty");
    }
    return tetraCentres;
}

DynRankView TetrahedronMeshManager::computeLocalStiffnessMatricesAndNablaPhi(
    Intrepid2::Basis<DeviceType, Scalar, Scalar> *basis,
    DynRankView const &cubPoints,
    DynRankView const &cubWeights,
    size_t numBasisFunctions,
    size_t numCubaturePoints)
{
    auto const numTetrahedrons{getNumTetrahedrons()};

    // 1. Reference basis grads
    DynRankView referenceBasisGrads("referenceBasisGrads", numBasisFunctions, numCubaturePoints, FEM_LIMITS_DEFAULT_SPACE_DIMENSION);
    basis->getValues(referenceBasisGrads, cubPoints, Intrepid2::OPERATOR_GRAD);

#ifdef USE_CUDA
    auto vertices{Kokkos::create_mirror_view_and_copy(MemorySpaceDevice(), m_tetraVertices)}; // From host to device.
#else
    DynRankView vertices(m_tetraVertices);
#endif

    // 2. Jacobians and measures
    DynRankView jacobians("jacobians", numTetrahedrons, numCubaturePoints, FEM_LIMITS_DEFAULT_SPACE_DIMENSION, FEM_LIMITS_DEFAULT_SPACE_DIMENSION);
    Intrepid2::CellTools<DeviceType>::setJacobian(jacobians, cubPoints, vertices, CellSelector::get(CellType::Tetrahedron));

    DynRankView invJacobians("invJacobians", numTetrahedrons, numCubaturePoints, FEM_LIMITS_DEFAULT_SPACE_DIMENSION, FEM_LIMITS_DEFAULT_SPACE_DIMENSION);
    Intrepid2::CellTools<DeviceType>::setJacobianInv(invJacobians, jacobians);

    DynRankView jacobiansDet("jacobiansDet", numTetrahedrons, numCubaturePoints);
    Intrepid2::CellTools<DeviceType>::setJacobianDet(jacobiansDet, jacobians);

    DynRankView cellMeasures("cellMeasures", numTetrahedrons, numCubaturePoints);
    Intrepid2::FunctionSpaceTools<DeviceType>::computeCellMeasure(cellMeasures, jacobiansDet, cubWeights);

    // 3. Transform reference grads
    DynRankView transformedBasisGradients("transformedBasisGradients", numTetrahedrons, numBasisFunctions, numCubaturePoints, FEM_LIMITS_DEFAULT_SPACE_DIMENSION);
    Intrepid2::FunctionSpaceTools<DeviceType>::HGRADtransformGRAD(transformedBasisGradients, invJacobians, referenceBasisGrads);

    // 4. Weighted basis grads
    DynRankView weightedBasisGrads("weightedBasisGrads", numTetrahedrons, numBasisFunctions, numCubaturePoints, FEM_LIMITS_DEFAULT_SPACE_DIMENSION);
    Intrepid2::FunctionSpaceTools<DeviceType>::multiplyMeasure(weightedBasisGrads, cellMeasures, transformedBasisGradients);

    // 5. Integrate local stiffness matrices
    DynRankView localStiffnessMatrices("localStiffnessMatrices", numTetrahedrons, numBasisFunctions, numBasisFunctions);
    Intrepid2::FunctionSpaceTools<DeviceType>::integrate(localStiffnessMatrices, weightedBasisGrads, transformedBasisGradients);

#ifdef USE_CUDA
    auto localStiffnessMatrices_host{Kokkos::create_mirror_view_and_copy(MemorySpaceHost(), localStiffnessMatrices)};
    auto weightedBasisGrads_host{Kokkos::create_mirror_view_and_copy(MemorySpaceHost(), weightedBasisGrads)};
#else
    auto &weightedBasisGrads_host{weightedBasisGrads};
#endif

    // 6. Assign nablaPhi
    for (size_t localTetraId{}; localTetraId < numTetrahedrons; ++localTetraId)
    {
        auto globalTetraId{getGlobalTetraId(localTetraId)};
        auto const &nodes{getTetrahedronNodes(localTetraId)};

        if (numBasisFunctions > nodes.size())
        {
            WARNINGMSG("Basis function count exceeds the number of nodes.");
        }

        for (short localNodeId{}; localNodeId < (short)numBasisFunctions; ++localNodeId)
        {
            auto globalNodeId{nodes[localNodeId].globalNodeId};
            Point grad(weightedBasisGrads_host(localTetraId, localNodeId, 0, 0),
                       weightedBasisGrads_host(localTetraId, localNodeId, 0, 1),
                       weightedBasisGrads_host(localTetraId, localNodeId, 0, 2));

            assignNablaPhi(globalTetraId, globalNodeId, grad);
        }
    }

    return localStiffnessMatrices;
}

void TetrahedronMeshManager::computeElectricFields()
{
    // We have a mapping of (Tetrahedron ID -> (Node ID -> Gradient of basis function)).
    // To obtain the electric field of the cell, we need to sum over all nodes the product
    // of the node's potential (φ_i) and the node's basis function gradient (∇φ_i).
    //
    // The formula is:
    // E_cell = Σ(φ_i * ∇φ_i)
    // where i is the global index of the node.
    for (auto &tetra : m_meshComponents)
    {
        ElectricField electricField{};
        bool missingData{};
        for (auto const &node : tetra.nodes)
        {
            if (node.potential && node.nablaPhi)
                electricField += ElectricField(node.nablaPhi->x(), node.nablaPhi->y(), node.nablaPhi->z()) * node.potential.value();
            else
                missingData = true;
        }

        if (missingData)
        {
            WARNINGMSG(util::stringify("Warning: Node potential or nablaPhi is not set for one or more nodes of tetrahedron ",
                                       tetra.globalTetraId, ". Those nodes are skipped in electric field computation.\n"));
        }

        tetra.electricField = Point(electricField.getX(), electricField.getY(), electricField.getZ());
    }
}
