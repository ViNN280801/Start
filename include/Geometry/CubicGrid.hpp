#ifndef CUBICGRID_HPP
#define CUBICGRID_HPP

#include <CGAL/Bbox_3.h>
#include <optional>

#include "DataHandling/TetrahedronMeshManager.hpp"
#include "Geometry/Mesh.hpp"
#include "Particle/Particle.hpp"
#include "ParticleInCellEngine/PICTypes.hpp"

/// @brief Represents a 3D grid index with x, y, and z components.
struct GridIndex
{
    short m_x{}, m_y{}, m_z{};

    GridIndex() {}
    GridIndex(short x, short y, short z) : m_x(x), m_y(y), m_z(z) {}
    bool operator==(GridIndex const &other) const { return m_x == other.m_x && m_y == other.m_y && m_z == other.m_z; }
    bool operator<(GridIndex const &other) const { return std::tie(m_x, m_y, m_z) < std::tie(other.m_x, other.m_y, other.m_z); }
};

/// @brief Represents a 3D grid that maps tetrahedrons to their containing cells.
class CubicGrid
{
private:
    std::map<size_t, std::vector<GridIndex>> m_tetrahedronCells; ///< Each tetrahedron contains list of cells. Key - ID of tetrahedron, Value - list of cells.
    double m_cubeEdgeSize{};                                     ///< Edge size of the cell.
    short m_divisionsX{}, m_divisionsY{}, m_divisionsZ{};        ///< Count of divisions by each axis.
    CGAL::Bbox_3 m_commonBbox;                                   ///< Common boundary box.
    TetrahedronMeshManager const &m_meshData;                    ///< Reference to TetrahedronMeshManager.

public:
    /**
     * @brief Ctor with params.
     * @param meshData Reference to TetrahedronMeshManager.
     * @param edgeSize Size of the cube edge.
     */
    CubicGrid(TetrahedronMeshManager const &meshData, double edgeSize);

    /**
     * @brief Getter for grid index by spatial position of some object.
     * @param x Position by X-axis.
     * @param y Position by Y-axis.
     * @param z Position by Z-axis.
     * @return Grid index.
     */
    GridIndex getGridIndexByPosition(double x, double y, double z) const;

    /**
     * @brief Getter for grid index by spatial position of some object.
     * @param point `CGAL::Point_3` point.
     * @return Grid index.
     */
    GridIndex getGridIndexByPosition(Point const &point) const;

    /**
     * @brief Checks if a given point is inside the grid.
     *
     * This method determines whether the specified point is within the bounds of the grid.
     * The grid is divided into cubic components, each of which may intersect with multiple tetrahedra.
     * The point is considered to be inside the grid if it is inside any of the tetrahedra within the cubic component.
     *
     * @param point The point to be checked.
     * @return true if the point is inside the grid, false otherwise.
     */
    bool isInsideTetrahedronMesh(Point const &point) const;

    /**
     * @brief Retrieves the list of tetrahedrons that intersect a specific grid cell.
     * @param index The index of the grid cell.
     * @return A vector of tetrahedrons that intersect with the specified cell.
     */
    std::vector<TetrahedronMeshManager::TetrahedronData> getTetrahedronsByGridIndex(GridIndex const &index) const;

    /// @return The total number of cells.
    constexpr size_t size() const { return static_cast<size_t>(m_divisionsX * m_divisionsY * m_divisionsZ); }

    /// @return The edge size of the cells.
    constexpr double edgeSize() const { return m_cubeEdgeSize; }

    /// @brief Prints a list of which tetrahedrons intersect with which grid cells.
    void printGrid() const;

    /**
     * @brief Finds the ID of the tetrahedron containing the given particle at a specified simulation time.
     *
     * This method searches for the tetrahedron that contains the given particle at a specified
     * simulation time using the provided particle tracker map. If no such tetrahedron is found,
     * the method returns `std::nullopt`.
     *
     * @param particleTracker A reference to the particle tracker map, which organizes particles
     *                        by simulation time and tetrahedron ID.
     * @param particle The particle to locate within the tracker.
     * @param timeMoment The simulation time at which to find the tetrahedron containing the particle.
     *                   Must be non-negative.
     * @return An optional containing the ID of the tetrahedron if the particle is found;
     *         otherwise, `std::nullopt`.
     *
     * @throws std::logic_error If the time moment is negative.
     * @throws std::invalid_argument If the particle tracker is empty or if the particle is invalid.
     * @throws std::runtime_error If the specified time moment does not exist in the particle tracker.
     */
    static std::optional<size_t> getContainingTetrahedron(ParticleTrackerMap const &particleTracker,
                                                          Particle const &particle,
                                                          double timeMoment);
};

#endif // !CUBICGRID_HPP
