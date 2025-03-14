#include "Geometry/Mesh/Surface/SurfaceMesh.hpp"
#include "Geometry/Basics/Edge.hpp"
#include "Utilities/GmshUtilities/GmshUtils.hpp"

void SurfaceMesh::_fillTriangles()
{
    // 1. Checking cell map on emptiness.
    if (m_triangleCellMap.empty())
        throw std::runtime_error("'m_triangleCellMap' is empty, failed to fill data member 'TriangleVector m_triangles'");

    // 2. Populate the triangle vector.
    for (auto const &[id, triangleCell] : m_triangleCellMap)
    {
        // 2.1. Skipping degenerate triangle.
        if (triangleCell.triangle.is_degenerate())
        {
            WARNINGMSG(util::stringify("Triangle with ID ", id, " is degenerate, skipping it..."));
            continue;
        }

        // 2.2. Adding non-degenerate triangle to triangles vector.
        m_triangles.emplace_back(triangleCell.triangle);
    }

    // 3. Checking triangles vector on emptiness after filling.
    if (m_triangles.empty())
        throw std::runtime_error("Cannot construct SurfaceMesh: all triangles are degenerate");
}

void SurfaceMesh::_constructAABB()
{
    // 1. Checking cell map on emptiness.
    if (m_triangleCellMap.empty())
        throw std::runtime_error("'m_triangleCellMap' is empty, failed to construct AABB tree for the surface mesh");

    // 2. Checking triangles vector on emptiness.
    if (m_triangles.empty())
        throw std::runtime_error("'m_triangles' is empty, failed to construct AABB tree for the surface mesh");

    // 3. Construct the AABB tree.
    m_aabbTree.insert(m_triangles.cbegin(), m_triangles.cend());
    m_aabbTree.build();

    // 4. Checking AABB tree on after filling it with triangles vector.
    if (m_aabbTree.empty())
        throw std::runtime_error("AABB Tree is empty after building. No valid triangles were inserted.");
}

void SurfaceMesh::_initialize()
{
    _fillTriangles();
    _constructAABB();
    _findNeighbors();
}

SurfaceMesh::SurfaceMesh(std::string_view meshFilename)
{
    m_triangleCellMap = GmshUtils::getTriangleCellsMap(meshFilename);
    _initialize();
}

SurfaceMesh::SurfaceMesh(TriangleCellMap_cref triangleCells)
{
    if (triangleCells.empty())
        throw std::invalid_argument("Cannot construct SurfaceMesh: triangle cell map is empty");
    m_triangleCellMap = triangleCells;
    _initialize();
}

SurfaceMesh::SurfaceMesh(std::string_view meshFilename, std::string_view physicalGroupName)
{
    if (!GmshUtils::hasPhysicalGroup(physicalGroupName, meshFilename))
        throw std::runtime_error(util::stringify("Mesh file '", meshFilename, "' does not contain physical group with name: ", physicalGroupName, '.'));

    m_triangleCellMap = GmshUtils::getCellsByPhysicalGroupName(physicalGroupName, meshFilename);
    _initialize();
}

size_t SurfaceMesh::getTotalCountOfSettledParticles() const noexcept
{
    return std::accumulate(m_triangleCellMap.cbegin(), m_triangleCellMap.cend(), size_t{}, [](size_t sum, auto const &entry)
                           { return sum + entry.second.count; });
}

void SurfaceMesh::_findNeighbors()
{
    // 1. Create a temporary edge map.
    std::unordered_map<Edge, std::vector<size_t>, EdgeHash> edgeMap;
    for (auto const &[id, cell] : m_triangleCellMap)
    {
        // 2. Extract the edges of the triangle.
        auto edges{getTriangleEdges(cell.triangle)};

        // 3. For each edge, add cellId to edgeMap.
        for (auto const &edge : edges)
            edgeMap[edge].push_back(id);
    }

    // 4. Find neighbors by common edges.
    for (auto &[id, cell] : m_triangleCellMap)
    {
        auto edges{getTriangleEdges(cell.triangle)};
        std::unordered_set<size_t> neighbors;

        // 5. For each edge, find neighbors.
        for (auto const &edge : edges)
        {
            if (auto it{edgeMap.find(edge)}; it != edgeMap.end())
            {
                for (auto neighbor_id : it->second)
                    if (neighbor_id != id)
                        neighbors.insert(neighbor_id);
            }
        }

        // 6. Save neighbors in the structure.
        cell.neighbor_ids.assign(neighbors.begin(), neighbors.end());
    }
}

std::vector<size_t> SurfaceMesh::getNeighborCells(size_t cellId) const
{
    if (auto it{m_triangleCellMap.find(cellId)}; it != m_triangleCellMap.end())
        return it->second.neighbor_ids;
    throw std::out_of_range("Cell ID not found");
}
