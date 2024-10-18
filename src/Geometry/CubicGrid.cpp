#include <algorithm>
#include <boost/dynamic_bitset.hpp>
#include <utility>
#include <vector>

#ifdef USE_OMP
#include <omp.h>
#endif

#if __cplusplus >= 202002L
#include <ranges>
#endif

#include "Geometry/CubicGrid.hpp"

CubicGrid::CubicGrid(TetrahedronMeshManager const &meshData, double edgeSize)
    : m_cubeEdgeSize(edgeSize), m_meshData(meshData)
{
    if (m_meshData.empty())
        return;

    // 1. Defining one common boundary box by merging all bboxes of tetrahedrons
    m_commonBbox = m_meshData.getMeshComponents().front().tetrahedron.bbox();
    for (const auto &tetrahedronData : m_meshData.getMeshComponents())
        m_commonBbox += tetrahedronData.tetrahedron.bbox();

    // 2. Calculating divisions for each axis
    m_divisionsX = static_cast<short>(std::ceil((m_commonBbox.xmax() - m_commonBbox.xmin()) / m_cubeEdgeSize));
    m_divisionsY = static_cast<short>(std::ceil((m_commonBbox.ymax() - m_commonBbox.ymin()) / m_cubeEdgeSize));
    m_divisionsZ = static_cast<short>(std::ceil((m_commonBbox.zmax() - m_commonBbox.zmin()) / m_cubeEdgeSize));

    // 3. Limitation on grid cells
    if (m_divisionsX * m_divisionsY * m_divisionsZ > MAX_GRID_SIZE)
        throw std::runtime_error("The grid is too small, you risk to overflow your memory with it");

        // 4. Mapping each tetrahedron with cells using parallelization
#ifdef USE_OMP
    std::vector<std::unordered_map<size_t, std::vector<GridIndex>>> localMaps(omp_get_max_threads());

#pragma omp parallel for
#endif
    for (size_t i = 0; i < m_meshData.getMeshComponents().size(); ++i)
    {
        const auto &tetrahedronData = m_meshData.getMeshComponents()[i];
        int threadId = 0;
#ifdef USE_OMP
        threadId = omp_get_thread_num();
#endif
        for (short x = 0; x < m_divisionsX; ++x)
        {
            for (short y = 0; y < m_divisionsY; ++y)
            {
                for (short z = 0; z < m_divisionsZ; ++z)
                {
                    CGAL::Bbox_3 cellBox(
                        m_commonBbox.xmin() + x * edgeSize,
                        m_commonBbox.ymin() + y * edgeSize,
                        m_commonBbox.zmin() + z * edgeSize,
                        m_commonBbox.xmin() + (x + 1) * edgeSize,
                        m_commonBbox.ymin() + (y + 1) * edgeSize,
                        m_commonBbox.zmin() + (z + 1) * edgeSize);

                    if (CGAL::do_overlap(cellBox, tetrahedronData.tetrahedron.bbox()))
                    {
#ifdef USE_OMP
                        localMaps[threadId][tetrahedronData.globalTetraId].emplace_back(x, y, z);
#else
                        m_tetrahedronCells[tetrahedronData.globalTetraId].emplace_back(x, y, z);
#endif
                    }
                }
            }
        }
    }

#ifdef USE_OMP
    // 5*. Merge the results from each thread-local map into the shared map
    for (const auto &localMap : localMaps)
        for (const auto &[tetrId, cells] : localMap)
            m_tetrahedronCells[tetrId].insert(m_tetrahedronCells[tetrId].end(), cells.begin(), cells.end());
#endif
}

GridIndex CubicGrid::getGridIndexByPosition(double x, double y, double z) const
{
    return {
        std::clamp(short((x - m_commonBbox.xmin()) / m_cubeEdgeSize), short(0), static_cast<short>(m_divisionsX - 1)),
        std::clamp(short((y - m_commonBbox.ymin()) / m_cubeEdgeSize), short(0), static_cast<short>(m_divisionsY - 1)),
        std::clamp(short((z - m_commonBbox.zmin()) / m_cubeEdgeSize), short(0), static_cast<short>(m_divisionsZ - 1))};
}

GridIndex CubicGrid::getGridIndexByPosition(Point const &point) const
{
    double x{CGAL_TO_DOUBLE(point.x())},
        y{CGAL_TO_DOUBLE(point.y())},
        z{CGAL_TO_DOUBLE(point.z())};
    return {
        std::clamp(short((x - m_commonBbox.xmin()) / m_cubeEdgeSize), short(0), static_cast<short>(m_divisionsX - 1)),
        std::clamp(short((y - m_commonBbox.ymin()) / m_cubeEdgeSize), short(0), static_cast<short>(m_divisionsY - 1)),
        std::clamp(short((z - m_commonBbox.zmin()) / m_cubeEdgeSize), short(0), static_cast<short>(m_divisionsZ - 1))};
}

bool CubicGrid::isInsideTetrahedronMesh(Point const &point) const
{
    auto gridIndex{getGridIndexByPosition(point)};
    auto tetrahedrons{getTetrahedronsByGridIndex(gridIndex)};

    // One cube grid component can return multiple tetrahedra, so we need to fill the vector of checkings with results of checkings.
    boost::dynamic_bitset<> checks(tetrahedrons.size());

#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < tetrahedrons.size(); ++i)
        checks[i] = Mesh::isPointInsideTetrahedron(point, tetrahedrons[i].tetrahedron);

    // If bitset contains at least one `true` - it means that point is inside the tetrahedron mesh.
    return checks.any();
}

std::vector<TetrahedronMeshManager::TetrahedronData> CubicGrid::getTetrahedronsByGridIndex(GridIndex const &index) const
{
    std::vector<TetrahedronMeshManager::TetrahedronData> meshParams;

#ifdef USE_OMP
    // To access m_tetrahedronCells in parallel, we use iterators and store them in a local vector first
    std::vector<std::pair<size_t, std::vector<GridIndex>>> tetrahedronCellsCopy;
    tetrahedronCellsCopy.reserve(m_tetrahedronCells.size());
    for (auto const &item : m_tetrahedronCells)
        tetrahedronCellsCopy.push_back(item);

#pragma omp parallel for shared(meshParams)
    for (size_t i = 0; i < tetrahedronCellsCopy.size(); ++i)
    {
        const auto &[tetrId, cells] = tetrahedronCellsCopy[i];
#else
    for (auto const &[tetrId, cells] : m_tetrahedronCells)
    {
#endif
#if __cplusplus >= 202002L
        if (std::ranges::find(cells.begin(), cells.end(), index) != cells.end())
#else
        if (std::find(cells.begin(), cells.end(), index) != cells.end())
#endif
        {
#ifdef USE_OMP
#pragma omp critical
#endif
            meshParams.push_back(m_meshData.getMeshDataByTetrahedronId(tetrId).value());
        }
    }
    return meshParams;
}

void CubicGrid::printGrid() const
{
    for (auto const &[id, cells] : m_tetrahedronCells)
    {
        std::cout << "Tetrahedron[" << id << "] is in cells: ";
        for (auto const &[x, y, z] : cells)
            std::cout << "[" << x << "][" << y << "][" << z << "] ";
        std::cout << std::endl;
    }
}
