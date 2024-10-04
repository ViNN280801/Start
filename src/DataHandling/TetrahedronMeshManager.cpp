#include <gmsh.h>
#include <memory>
#include <mpi.h>
#include <mutex>

#include "DataHandling/TetrahedronMeshManager.hpp"

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

TetrahedronMeshManager::TetrahedronMeshManager(std::string_view mesh_filename)
{
    util::check_gmsh_mesh_file(mesh_filename);

    int rank = 0, size = 1;
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, std::addressof(rank));
    MPI_Comm_size(MPI_COMM_WORLD, std::addressof(size));
#endif

    std::vector<size_t> localElementTags;
    std::vector<size_t> localNodeTags;
    std::map<size_t, std::array<double, 3>> localNodeCoordinates;

    if (rank == 0)
    {
        try
        {
            gmsh::open(mesh_filename.data());

            // Read nodes
            std::vector<std::size_t> nodeTags;
            std::vector<double> coord, parametricCoord;
            gmsh::model::mesh::getNodes(nodeTags, coord, parametricCoord, -1, -1, false, false);

            // Map node IDs to coordinates
            std::map<size_t, std::array<double, 3>> nodeCoordinatesMap;
            for (size_t i = 0; i < nodeTags.size(); ++i)
            {
                size_t nodeID = nodeTags[i];
                std::array<double, 3> coords = {coord[i * 3 + 0], coord[i * 3 + 1], coord[i * 3 + 2]};
                nodeCoordinatesMap[nodeID] = coords;
            }

            // Read tetrahedron elements
            std::vector<size_t> elTags, nodeTagsByEl;
            gmsh::model::mesh::getElementsByType(4, elTags, nodeTagsByEl, -1);

            // Partition elements among MPI processes
            size_t numElements = elTags.size();
            std::vector<std::vector<size_t>> elementsPerProc(size);
            std::vector<std::vector<size_t>> nodeTagsPerProc(size);

            for (size_t i = 0; i < numElements; ++i)
            {
                int destRank = i % size;
                elementsPerProc[destRank].push_back(elTags[i]);
                for (int j = 0; j < 4; ++j)
                {
                    nodeTagsPerProc[destRank].push_back(nodeTagsByEl[i * 4 + j]);
                }
            }

            // Send partitioned mesh data to other processes
            for (int destRank = 1; destRank < size; ++destRank)
            {
                // Send the number of elements
                size_t numLocalElements = elementsPerProc[destRank].size();
                MPI_Send(&numLocalElements, 1, MPI_UNSIGNED_LONG, destRank, 0, MPI_COMM_WORLD);

                // Send element IDs
                MPI_Send(elementsPerProc[destRank].data(), numLocalElements, MPI_UNSIGNED_LONG, destRank, 1, MPI_COMM_WORLD);

                // Send node IDs
                size_t numLocalNodes = nodeTagsPerProc[destRank].size();
                MPI_Send(&numLocalNodes, 1, MPI_UNSIGNED_LONG, destRank, 2, MPI_COMM_WORLD);
                MPI_Send(nodeTagsPerProc[destRank].data(), numLocalNodes, MPI_UNSIGNED_LONG, destRank, 3, MPI_COMM_WORLD);

                // Collect coordinates for the nodes used
                std::vector<double> localNodeCoords;
                for (size_t nodeID : nodeTagsPerProc[destRank])
                {
                    auto coords = nodeCoordinatesMap[nodeID];
                    localNodeCoords.insert(localNodeCoords.end(), coords.begin(), coords.end());
                }

                // Send node coordinates
                MPI_Send(localNodeCoords.data(), localNodeCoords.size(), MPI_DOUBLE, destRank, 4, MPI_COMM_WORLD);
            }

            // Root process keeps its part of the data
            localElementTags = elementsPerProc[0];
            localNodeTags = nodeTagsPerProc[0];

            for (size_t nodeID : localNodeTags)
            {
                localNodeCoordinates[nodeID] = nodeCoordinatesMap[nodeID];
            }

            gmsh::finalize();
        }
        catch (std::exception const &e)
        {
            throw std::runtime_error(util::stringify("Error initializing TetrahedronMeshManager: ", e.what()));
        }
    }
    else
    {
        // Receive mesh data from the root process
        // Receive the number of elements
        size_t numLocalElements;
        MPI_Recv(&numLocalElements, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Receive element IDs
        localElementTags.resize(numLocalElements);
        MPI_Recv(localElementTags.data(), numLocalElements, MPI_UNSIGNED_LONG, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Receive node IDs
        size_t numLocalNodes;
        MPI_Recv(&numLocalNodes, 1, MPI_UNSIGNED_LONG, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        localNodeTags.resize(numLocalNodes);
        MPI_Recv(localNodeTags.data(), numLocalNodes, MPI_UNSIGNED_LONG, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Receive node coordinates
        std::vector<double> localNodeCoords(numLocalNodes * 3);
        MPI_Recv(localNodeCoords.data(), localNodeCoords.size(), MPI_DOUBLE, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Reconstruct nodeCoordinatesMap
        for (size_t i = 0; i < numLocalNodes; ++i)
        {
            size_t nodeID = localNodeTags[i];
            std::array<double, 3> coords = {
                localNodeCoords[i * 3 + 0],
                localNodeCoords[i * 3 + 1],
                localNodeCoords[i * 3 + 2]};
            localNodeCoordinates[nodeID] = coords;
        }
    }

    // Now use localElementTags and localNodeCoordinates to construct tetrahedrons and store them in m_meshComponents
    for (size_t tetrahedronID : localElementTags)
    {
        std::array<Point, 4> vertices;
        std::array<size_t, 4> nodes;
        for (size_t j = 0; j < 4; ++j)
        {
            size_t nodeID = localNodeTags[j];
            auto coords = localNodeCoordinates.at(nodeID);
            vertices[j] = Point(coords[0], coords[1], coords[2]);
            nodes[j] = nodeID;
        }

        Tetrahedron tetrahedron(vertices[0], vertices[1], vertices[2], vertices[3]);
        TetrahedronData data{tetrahedronID, tetrahedron, {}, std::nullopt};

        for (size_t j = 0; j < nodes.size(); ++j)
            data.nodes[j] = {nodes[j], vertices[j], std::nullopt, std::nullopt};

        m_meshComponents.emplace_back(data);
    }
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
    auto it{std::ranges::find_if(m_meshComponents, [globalTetrahedronId](const TetrahedronData &data)
                                 { return data.globalTetraId == globalTetrahedronId; })};
    if (it != m_meshComponents.cend())
        return *it;
    return std::nullopt;
}

void TetrahedronMeshManager::assignNablaPhi(size_t tetrahedronId, size_t nodeId, Point const &gradient)
{
    auto it{std::ranges::find_if(m_meshComponents, [tetrahedronId](const TetrahedronData &data)
                                 { return data.globalTetraId == tetrahedronId; })};
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
    auto it{std::ranges::find_if(m_meshComponents, [tetrahedronId](const TetrahedronData &data)
                                 { return data.globalTetraId == tetrahedronId; })};
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
        WARNINGMSG("Tetrahedron - nodes map is empty");
    return tetrahedronNodesMap;
}

std::map<size_t, std::vector<size_t>> TetrahedronMeshManager::getNodeTetrahedronsMap() const
{
    std::map<size_t, std::vector<size_t>> nodeTetrahedronsMap;

    for (auto const &meshData : m_meshComponents)
        for (short i{}; i < 4; ++i)
            nodeTetrahedronsMap[meshData.nodes.at(i).globalNodeId].emplace_back(meshData.globalTetraId);

    if (nodeTetrahedronsMap.empty())
        WARNINGMSG("Node - tetrahedrons map is empty");
    return nodeTetrahedronsMap;
}

std::map<size_t, Point> TetrahedronMeshManager::getTetrahedronCenters() const
{
    std::map<size_t, Point> tetraCentres;

    for (auto const &meshData : m_meshComponents)
        tetraCentres[meshData.globalTetraId] = meshData.getTetrahedronCenter();

    if (tetraCentres.empty())
        WARNINGMSG("Tetrahedron centres map is empty");
    return tetraCentres;
}
