#include "Geometry/Mesh.hpp"
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
    m_aabbTree.insert(m_triangles.begin(), m_triangles.end());
    m_aabbTree.build();

    // 4. Checking AABB tree on after filling it with triangles vector.
    if (m_aabbTree.empty())
        throw std::runtime_error("AABB Tree is empty after building. No valid triangles were inserted.");
}

void SurfaceMesh::_initialize()
{
    _fillTriangles();
    _constructAABB();
}

SurfaceMesh::SurfaceMesh(std::string_view meshFilename)
{
    m_triangleCellMap = Mesh::getMeshParams(meshFilename);
    _initialize();
}

SurfaceMesh::SurfaceMesh(TriangleCellMap const &triangleCells)
{
    if (triangleCells.empty())
        throw std::invalid_argument("Cannot construct SurfaceMesh: triangle cell map is empty");
    m_triangleCellMap = triangleCells;
    _initialize();
}

SurfaceMesh::SurfaceMesh(std::string_view meshFilename, std::string_view physicalGroupName)
{
    m_triangleCellMap = GmshUtils::getCellsByPhysicalGroupName(physicalGroupName, meshFilename);
    _initialize();
}
