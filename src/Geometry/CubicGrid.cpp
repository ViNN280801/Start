#include <algorithm>
#include <boost/dynamic_bitset.hpp>
#include <utility>
#include <vector>

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

    try
    {
        // 3. Mapping each tetrahedron with cells using parallelization
        for (size_t i{}; i < m_meshData.getMeshComponents().size(); ++i)
        {
            const auto &tetrahedronData = m_meshData.getMeshComponents()[i];
            for (short x{}; x < m_divisionsX; ++x)
            {
                for (short y{}; y < m_divisionsY; ++y)
                {
                    for (short z{}; z < m_divisionsZ; ++z)
                    {
                        CGAL::Bbox_3 cellBox(
                            m_commonBbox.xmin() + x * edgeSize,
                            m_commonBbox.ymin() + y * edgeSize,
                            m_commonBbox.zmin() + z * edgeSize,
                            m_commonBbox.xmin() + (x + 1) * edgeSize,
                            m_commonBbox.ymin() + (y + 1) * edgeSize,
                            m_commonBbox.zmin() + (z + 1) * edgeSize);

                        if (CGAL::do_overlap(cellBox, tetrahedronData.tetrahedron.bbox()))
                            m_tetrahedronCells[tetrahedronData.globalTetraId].emplace_back(x, y, z);
                    }
                }
            }
        }
    }
    catch (std::exception const &ex)
    {
        WARNINGMSG(util::stringify("Error while constructing cubic grid: ", ex.what()));
    }
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
    return {
        std::clamp(short((point.x() - m_commonBbox.xmin()) / m_cubeEdgeSize), short(0), static_cast<short>(m_divisionsX - 1)),
        std::clamp(short((point.y() - m_commonBbox.ymin()) / m_cubeEdgeSize), short(0), static_cast<short>(m_divisionsY - 1)),
        std::clamp(short((point.z() - m_commonBbox.zmin()) / m_cubeEdgeSize), short(0), static_cast<short>(m_divisionsZ - 1))};
}

bool CubicGrid::isInsideTetrahedronMesh(Point const &point) const
{
    auto gridIndex{getGridIndexByPosition(point)};
    auto tetrahedrons{getTetrahedronsByGridIndex(gridIndex)};

    // One cube grid component can return multiple tetrahedra, so we need to fill the vector of checkings with results of checkings.
    boost::dynamic_bitset<> checks(tetrahedrons.size());

    for (size_t i{}; i < tetrahedrons.size(); ++i)
        checks[i] = Mesh::isPointInsideTetrahedron(point, tetrahedrons[i].tetrahedron);

    // If bitset contains at least one `true` - it means that point is inside the tetrahedron mesh.
    return checks.any();
}

std::vector<TetrahedronMeshManager::TetrahedronData> CubicGrid::getTetrahedronsByGridIndex(GridIndex const &index) const
{
    std::vector<TetrahedronMeshManager::TetrahedronData> meshParams;

    for (auto const &[tetrId, cells] : m_tetrahedronCells)
    {
#if __cplusplus >= 202002L
        if (std::ranges::find(cells.begin(), cells.end(), index) != cells.end())
#else
        if (std::find(cells.begin(), cells.end(), index) != cells.end())
#endif
        {
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

std::optional<size_t> CubicGrid::getContainingTetrahedron(ParticleTrackerMap const &particleTracker,
                                                          Particle const &particle,
                                                          double timeMoment)
#ifndef USE_SERIAL
    noexcept
#endif
{
    auto particleInTetrahedronMapIter{particleTracker.find(timeMoment)}; // std::map<size_t, ParticleVector>, where size_t - ID of the tetrahedron.
    if (particleInTetrahedronMapIter == particleTracker.end())
        return std::nullopt;

#ifdef USE_SERIAL
    if (timeMoment < 0)
        throw std::logic_error("Time moment can't be negative, but you passed: " + std::to_string(timeMoment));
    if (particleTracker.empty())
        throw std::invalid_argument("Particle tracker map is empty. Cannot search for tetrahedrons.");
#else
    if (timeMoment < 0)
    {
        WARNINGMSG(util::stringify("Time moment can't be negative, but you passed: ", timeMoment));
        return std::nullopt;
    }
    if (particleTracker.empty())
    {
        WARNINGMSG("Particle tracker map is empty. Cannot search for tetrahedrons.");
        return std::nullopt;
    }
#endif

    for (auto const &[tetraId, particlesInside] : particleInTetrahedronMapIter->second)
    {
#if __cplusplus >= 202002L
        if (std::ranges::find_if(particlesInside, [&particle](Particle const &storedParticle)
                                 { return particle.getId() == storedParticle.getId(); }) != particlesInside.cend())
#else
        if (std::find_if(particlesInside.begin(), particlesInside.end(), [&particle](Particle const &storedParticle)
                         { return particle.getId() == storedParticle.getId(); }) != particlesInside.cend())
#endif
            return tetraId;
    }
    return std::nullopt;
}
