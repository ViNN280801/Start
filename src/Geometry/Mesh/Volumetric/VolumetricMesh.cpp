#include "Geometry/Mesh/Volumetric/VolumetricMesh.hpp"
#include "FiniteElementMethod/Cell/CellSelector.hpp"
#include "FiniteElementMethod/Utils/FEMLimits.hpp"
#include "Geometry/Basics/GeometryVector.hpp"
#include "Utilities/GmshUtilities/GmshUtils.hpp"

Point VolumetricMesh::TetrahedronData::getTetrahedronCenter() const
{
    if (m_nodes.empty())
        throw std::runtime_error("Tetrahedron has no nodes, check the mesh file.");

    double x{}, y{}, z{};
    for (auto const &node : m_nodes)
    {
        x += node.m_nodeCoords.x();
        y += node.m_nodeCoords.y();
        z += node.m_nodeCoords.z();
    }
    return Point{x / 4.0, y / 4.0, z / 4.0};
}

bool VolumetricMesh::TetrahedronData::isPointInside(Point_cref point) const noexcept
{
    CGAL::Oriented_side oriented_side{m_tetrahedron.oriented_side(point)};
    if (oriented_side == CGAL::ON_POSITIVE_SIDE)
        return true;
    else if (oriented_side == CGAL::ON_NEGATIVE_SIDE)
        return false;
    else
        // TODO: Correctly handle case when particle is on boundary of tetrahedron.
        return true;
}

void VolumetricMesh::_readMesh(std::string_view mesh_filename)
{
    try
    {
        // Initialize Gmsh and open the mesh file
        GmshUtils::checkAndOpenMesh(mesh_filename);

        // Read all node data
        std::vector<std::size_t> nodeTags;
        std::vector<double> coord, parametricCoord;
        gmsh::model::mesh::getNodes(nodeTags, coord, parametricCoord);

        // Map node IDs to their coordinates
        std::map<size_t, std::array<double, 3>> nodeCoordinatesMap;
        for (size_t i{}; i < nodeTags.size(); ++i)
        {
            size_t nodeID{nodeTags[i]};
            std::array<double, 3> coords = {coord[i * 3 + 0], coord[i * 3 + 1], coord[i * 3 + 2]};
            nodeCoordinatesMap[nodeID] = coords;
        }

        // Read tetrahedron elements (element type 4 corresponds to tetrahedra in GMSH)
        std::vector<size_t> elTags, nodeTagsByEl;
        gmsh::model::mesh::getElementsByType(4, elTags, nodeTagsByEl);

        // Clear any existing mesh data
        m_meshComponents.clear();
        m_globalToLocalTetraId.clear();

        // Process each tetrahedron
        for (size_t i{}; i < elTags.size(); ++i)
        {
            size_t tetrahedronID{elTags[i]};
            std::array<size_t, 4> nodeIDs;

            // Get the four node IDs for this tetrahedron
            for (short j{}; j < 4; ++j)
                nodeIDs[j] = nodeTagsByEl[i * 4 + j];

            // Create vertex points from node coordinates
            std::array<Point, 4> vertices;
            for (short j{}; j < 4; ++j)
            {
                size_t nodeID = nodeIDs[j];
                auto coords = nodeCoordinatesMap.at(nodeID);
                vertices[j] = Point(coords[0], coords[1], coords[2]);
            }

            // Create the tetrahedron
            Tetrahedron tetrahedron(vertices[0], vertices[1], vertices[2], vertices[3]);
            TetrahedronData data{tetrahedronID, tetrahedron, {}, std::nullopt};

            // Set up node data for the tetrahedron
            for (short j{}; j < 4; ++j)
                data.m_nodes[j] = {nodeIDs[j], vertices[j], std::nullopt, std::nullopt};

            // Add to mesh components and update mapping
            m_meshComponents.emplace_back(data);
            m_globalToLocalTetraId[tetrahedronID] = m_meshComponents.size() - 1;
        }

        // Setup DynRankViewHost for vertices (used by FEM computations)
        m_tetraVertices = DynRankViewHost("tetraVertices", elTags.size(), 4, 3);
        m_tetraGlobalIds = DynRankViewHost("tetraGlobalIds", elTags.size());

        for (size_t i{}; i < m_meshComponents.size(); ++i)
        {
            auto const &tetra = m_meshComponents[i];
            m_tetraGlobalIds(i) = tetra.m_globalTetraId;

            for (short j{}; j < 4; ++j)
            {
                m_tetraVertices(i, j, 0) = tetra.m_nodes[j].m_nodeCoords.x();
                m_tetraVertices(i, j, 1) = tetra.m_nodes[j].m_nodeCoords.y();
                m_tetraVertices(i, j, 2) = tetra.m_nodes[j].m_nodeCoords.z();
            }
        }

        LOGMSG(util::stringify("Successfully loaded mesh with ", m_meshComponents.size(),
                               " tetrahedra and ", nodeCoordinatesMap.size(), " nodes"));
    }
    catch (std::exception const &e)
    {
        throw std::runtime_error(util::stringify("Error in _readMesh: ", e.what()));
    }
}

void VolumetricMesh::_buildViews()
{
    m_tetraVertices = DynRankViewHost("tetraVertices", getNumTetrahedrons(), 4, 3);
    m_tetraGlobalIds = DynRankViewHost("tetraGlobalIds", getNumTetrahedrons());

    for (size_t localId{}; localId < getNumTetrahedrons(); ++localId)
    {
        auto const &data{m_meshComponents[localId]};
        m_tetraGlobalIds(localId) = data.m_globalTetraId;
        for (short node{}; node < 4; ++node)
        {
            m_tetraVertices(localId, node, 0) = data.m_nodes[node].m_nodeCoords.x();
            m_tetraVertices(localId, node, 1) = data.m_nodes[node].m_nodeCoords.y();
            m_tetraVertices(localId, node, 2) = data.m_nodes[node].m_nodeCoords.z();
        }
    }

    m_globalToLocalTetraId.reserve(getNumTetrahedrons());
    for (size_t localId{}; localId < getNumTetrahedrons(); ++localId)
        m_globalToLocalTetraId[m_tetraGlobalIds(localId)] = localId;
}

VolumetricMesh::VolumetricMesh(std::string_view mesh_filename)
{
    GmshUtils::checkGmshMeshFile(mesh_filename);
    _readMesh(mesh_filename);
    _buildViews();
}

void VolumetricMesh::print() const noexcept
{
    for (auto const &meshComponent : m_meshComponents)
    {
        std::cout << "Tetrahedron[" << meshComponent.m_globalTetraId << "]\n";

        for (short i{}; i < 4; ++i)
        {
            std::cout << "Vertex[" << meshComponent.m_nodes.at(i).m_globalNodeId << "]: ("
                      << meshComponent.m_nodes.at(i).m_nodeCoords.x() << ", "
                      << meshComponent.m_nodes.at(i).m_nodeCoords.y() << ", "
                      << meshComponent.m_nodes.at(i).m_nodeCoords.z() << ")\n";

            if (meshComponent.m_nodes.at(i).m_nablaPhi)
            {
                std::cout << "  ∇φ: ("
                          << meshComponent.m_nodes.at(i).m_nablaPhi->x() << ", "
                          << meshComponent.m_nodes.at(i).m_nablaPhi->y() << ", "
                          << meshComponent.m_nodes.at(i).m_nablaPhi->z() << ")\n";
            }
            else
            {
                std::cout << "  ∇φ: empty\n";
            }

            if (meshComponent.m_nodes.at(i).m_potential)
            {
                std::cout << "  Potential φ: " << meshComponent.m_nodes.at(i).m_potential.value() << "\n";
            }
            else
            {
                std::cout << "  Potential φ: empty\n";
            }
        }

        if (meshComponent.m_electricField)
        {
            std::cout << "ElectricField: ("
                      << meshComponent.m_electricField->x() << ", "
                      << meshComponent.m_electricField->y() << ", "
                      << meshComponent.m_electricField->z() << ")\n";
        }
        else
        {
            std::cout << "ElectricField: empty\n";
        }
    }

    std::cout << std::endl;
}

std::optional<VolumetricMesh::TetrahedronData> VolumetricMesh::getMeshDataByTetrahedronId(size_t globalTetrahedronId) const
{
#if __cplusplus >= 202002L
    auto it{std::ranges::find_if(m_meshComponents, [globalTetrahedronId](const TetrahedronData &data)
                                 { return data.m_globalTetraId == globalTetrahedronId; })};
#else
    auto it{std::find_if(m_meshComponents.cbegin(), m_meshComponents.cend(), [globalTetrahedronId](const TetrahedronData &data)
                         { return data.m_globalTetraId == globalTetrahedronId; })};
#endif
    if (it != m_meshComponents.cend())
        return *it;
    return std::nullopt;
}

void VolumetricMesh::assignNablaPhi(size_t tetrahedronId, size_t nodeId, Point const &gradient)
{
#if __cplusplus >= 202002L
    auto it{std::ranges::find_if(m_meshComponents, [tetrahedronId](const TetrahedronData &data)
                                 { return data.m_globalTetraId == tetrahedronId; })};
#else
    auto it{std::find_if(m_meshComponents.begin(), m_meshComponents.end(), [tetrahedronId](const TetrahedronData &data)
                         { return data.m_globalTetraId == tetrahedronId; })};
#endif
    if (it != m_meshComponents.cend())
    {
        for (auto &node : it->m_nodes)
        {
            if (node.m_globalNodeId == nodeId)
            {
                node.m_nablaPhi = gradient;
                return;
            }
        }
    }
}

void VolumetricMesh::assignPotential(size_t nodeId, double potential)
{
    for (auto &tetrahedron : m_meshComponents)
        for (auto &node : tetrahedron.m_nodes)
            if (node.m_globalNodeId == nodeId)
                node.m_potential = potential;
}

void VolumetricMesh::assignElectricField(size_t tetrahedronId, Point const &electricField)
{
#if __cplusplus >= 202002L
    auto it{std::ranges::find_if(m_meshComponents, [tetrahedronId](const TetrahedronData &data)
                                 { return data.m_globalTetraId == tetrahedronId; })};
#else
    auto it{std::find_if(m_meshComponents.begin(), m_meshComponents.end(), [tetrahedronId](const TetrahedronData &data)
                         { return data.m_globalTetraId == tetrahedronId; })};
#endif
    if (it != m_meshComponents.end())
    {
        it->m_electricField = electricField;
    }
}

std::map<size_t, std::vector<size_t>> VolumetricMesh::getTetrahedronNodesMap() const
{
    std::map<size_t, std::vector<size_t>> tetrahedronNodesMap;

    for (auto const &meshData : m_meshComponents)
        for (short i{}; i < 4; ++i)
            tetrahedronNodesMap[meshData.m_globalTetraId].emplace_back(meshData.m_nodes.at(i).m_globalNodeId);

    if (tetrahedronNodesMap.empty())
    {
        WARNINGMSG("Tetrahedron - nodes map is empty");
    }
    return tetrahedronNodesMap;
}

std::map<size_t, std::vector<size_t>> VolumetricMesh::getNodeTetrahedronsMap() const
{
    std::map<size_t, std::vector<size_t>> nodeTetrahedronsMap;

    for (auto const &meshData : m_meshComponents)
        for (short i{}; i < 4; ++i)
            nodeTetrahedronsMap[meshData.m_nodes.at(i).m_globalNodeId].emplace_back(meshData.m_globalTetraId);

    if (nodeTetrahedronsMap.empty())
    {
        WARNINGMSG("Node - tetrahedrons map is empty");
    }
    return nodeTetrahedronsMap;
}

std::map<size_t, Point> VolumetricMesh::getTetrahedronCenters() const
{
    std::map<size_t, Point> tetraCentres;

    for (auto const &meshData : m_meshComponents)
        tetraCentres[meshData.m_globalTetraId] = meshData.getTetrahedronCenter();

    if (tetraCentres.empty())
    {
        WARNINGMSG("Tetrahedron centres map is empty");
    }
    return tetraCentres;
}

DynRankView VolumetricMesh::computeLocalStiffnessMatricesAndNablaPhi(
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
            auto globalNodeId{nodes[localNodeId].m_globalNodeId};
            Point grad(weightedBasisGrads_host(localTetraId, localNodeId, 0, 0),
                       weightedBasisGrads_host(localTetraId, localNodeId, 0, 1),
                       weightedBasisGrads_host(localTetraId, localNodeId, 0, 2));

            assignNablaPhi(globalTetraId, globalNodeId, grad);
        }
    }

    return localStiffnessMatrices;
}

void VolumetricMesh::computeElectricFields()
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
        for (auto const &node : tetra.m_nodes)
        {
            if (node.m_potential && node.m_nablaPhi)
                electricField += ElectricField(node.m_nablaPhi->x(), node.m_nablaPhi->y(), node.m_nablaPhi->z()) * node.m_potential.value();
            else
                missingData = true;
        }

        if (missingData)
        {
            WARNINGMSG(util::stringify("Warning: Node potential or nablaPhi is not set for one or more nodes of tetrahedron ",
                                       tetra.m_globalTetraId, ". Those nodes are skipped in electric field computation.\n"));
        }

        tetra.m_electricField = Point(electricField.getX(), electricField.getY(), electricField.getZ());
    }
}
