#ifndef SURFACE_MESH_HPP
#define SURFACE_MESH_HPP

#include "Geometry/Mesh/Surface/AABBTree/AABBTree.hpp"
#include "Geometry/Mesh/Surface/TriangleCell.hpp"

/**
 * @class SurfaceMesh
 * @brief Represents a surface mesh constructed from triangular cells.
 *
 * The `SurfaceMesh` class manages a collection of triangular cells that form a surface.
 * It provides functionalities for:
 * - Storing and accessing the triangle cell data.
 * - Constructing an **Axis-Aligned Bounding Box (AABB) tree** for efficient spatial queries.
 * - Extracting triangle cells from a given mesh file or a specified physical group.
 *
 * @details
 * **Key Features:**
 * - Uses **Gmsh** to retrieve mesh data and extract triangular cells.
 * - Supports **AABB trees** (using CGAL) for efficient geometric queries.
 * - Provides constructors for loading a surface mesh from:
 *   - A `.msh` file.
 *   - A specific **physical group** in the mesh.
 *   - A manually provided `TriangleCellMap` (for advanced use cases).
 *
 * **Internal Methods:**
 * - `_fillTriangles()`: Filters degenerate triangles and stores valid ones.
 * - `_constructAABB()`: Builds an AABB tree from the stored triangles.
 * - `_initialize()`: Calls the above methods to construct a valid mesh.
 *
 * @note
 * - The **AABB tree** improves performance when performing **nearest-neighbor searches** or **ray-intersection queries**.
 * - Degenerate (zero-area) triangles are automatically skipped and logged as warnings.
 * - This class is designed to handle **surface meshes** only, not volumetric meshes.
 *
 * **Example Usage:**
 * @code
 * // Load a surface mesh from a .msh file
 * SurfaceMesh surfaceMesh("example.msh");
 *
 * // Retrieve the AABB tree for efficient spatial queries
 * auto const& tree = surfaceMesh.getAABBTree();
 *
 * // Iterate over all stored triangles
 * for (const auto& triangle : surfaceMesh.getTriangles())
 * {
 *     std::cout << triangle << std::endl;
 * }
 * @endcode
 */
class SurfaceMesh
{
private:
    TriangleVector m_triangles;        ///< Stores non-degenerate triangles for the surface mesh.
    TriangleCellMap m_triangleCellMap; ///< Maps triangle IDs to triangle cells containing geometric and simulation data.
    AABBTree m_aabbTree;               ///< AABB tree for efficient spatial queries.

    /**
     * @brief Fills the triangle vector with valid, non-degenerate triangles.
     *
     * This method iterates over `m_triangleCellMap`, extracting valid triangles and
     * filtering out **degenerate** ones (triangles with zero area).
     * - Degenerate triangles are **logged** and **skipped**.
     * - If all triangles are degenerate, an exception is thrown.
     *
     * @throw GeometryTriangleCellMapEmptyException If `m_triangleCellMap` is empty.
     * @throw GeometryTrianglesVectorEmptyException If no valid triangles are found after filtering.
     */
    void _fillTriangles();

    /**
     * @brief Constructs an AABB tree from the stored triangle data.
     *
     * This method builds an **Axis-Aligned Bounding Box (AABB) tree**, allowing efficient
     * spatial operations such as:
     * - Nearest-neighbor searches.
     * - Fast intersection tests.
     *
     * The AABB tree is populated using `m_triangles`, which must be pre-filled.
     * - If `m_triangles` is empty, an exception is thrown.
     * - If the constructed AABB tree is empty after insertion, an exception is thrown.
     *
     * @throw GeometryTrianglesVectorEmptyException If `m_triangles` is empty.
     * @throw GeometryAABBTreeEmptyException If the AABB tree cannot be built.
     */
    void _constructAABB();

    /**
     * @brief Finds and sets neighboring cells for all triangles
     * @details Two cells are considered neighbors if they share an edge
     */
    void _findNeighbors();

    /**
     * @brief Initializes the surface mesh by populating triangles and constructing the AABB tree.
     *
     * This method calls `_fillTriangles()` and `_constructAABB()` in sequence, ensuring that:
     * - The triangle vector is correctly populated.
     * - A valid AABB tree is constructed.
     */
    void _initialize();

public:
    SurfaceMesh() = default;

    /**
     * @brief Constructs a SurfaceMesh from a given mesh file.
     *
     * This constructor reads a `.msh` file and extracts all triangular cells present in the mesh.
     * - Initializes the **AABB tree** for spatial queries.
     *
     * @param meshFilename The path to the Gmsh `.msh` file.
     * @throw GeometryInitializationException If the file is missing, invalid, or contains no valid triangles.
     */
    explicit SurfaceMesh(std::string_view meshFilename);

    /**
     * @brief Constructs a SurfaceMesh from an existing `TriangleCellMap`.
     *
     * This constructor allows **manual injection** of triangle data, bypassing Gmsh loading.
     * - Useful for advanced cases where a custom triangle dataset is used.
     * - Automatically filters degenerate triangles and builds the **AABB tree**.
     *
     * @param triangleCells A `TriangleCellMap` containing triangle definitions.
     * @throw GeometryTriangleCellMapEmptyException If `triangleCells` is empty.
     */
    explicit SurfaceMesh(TriangleCellMap_cref triangleCells);

    /**
     * @brief Constructs a SurfaceMesh from a specific physical group in a mesh file.
     *
     * This constructor extracts **only** the triangles associated with a specified **physical group**.
     * - Uses `GmshUtils::getCellsByPhysicalGroupName()` to retrieve triangle data.
     * - Builds the **AABB tree** after loading.
     *
     * @param meshFilename The path to the Gmsh `.msh` file.
     * @param physicalGroupName The name of the physical group containing the desired triangles.
     * @throw GeometryPhysicalGroupNotFoundException If the physical group does not exist or contains no valid triangles.
     */
    explicit SurfaceMesh(std::string_view meshFilename, std::string_view physicalGroupName);

    /**
     * @brief Provides read-only access to the AABB tree.
     * @return const AABB_Tree_Triangle& Reference to the AABB tree for spatial queries.
     *
     * The AABB tree enables **fast geometric operations**, such as:
     * - **Ray intersections** (useful for ray-tracing simulations).
     * - **Proximity checks** (e.g., finding the nearest triangle to a given point).
     */
    AABBTree_cref getAABBTree() const { return m_aabbTree; }

    /**
     * @brief Provides read-only access to the stored triangle vector.
     * @return const TriangleVector& Reference to the vector of non-degenerate triangles.
     *
     * **Note:**
     * - This vector only contains **valid (non-degenerate)** triangles.
     * - It is primarily used for **rendering** or **direct triangle iteration**.
     */
    constexpr TriangleVector_cref getTriangles() const { return m_triangles; }

    /**
     * @brief Provides read-only access to the triangle cell map.
     * @return const TriangleCellMap& Reference to the triangle cell map.
     *
     * The **triangle cell map** allows direct access to:
     * - Triangle geometry.
     * - Triangle area.
     * - Particle counts (if applicable in a simulation).
     */
    constexpr TriangleCellMap_cref getTriangleCellMap() const { return m_triangleCellMap; }

    /**
     * @brief Provides read/write access to the triangle cell map.
     * @return TriangleCellMap& Reference to the triangle cell map.
     *
     * Allows modification of the triangle data, but changes may require **rebuilding the AABB tree**.
     * - Modifications to triangle geometry should be followed by `_constructAABB()`.
     */
    constexpr TriangleCellMap_ref getTriangleCellMap() { return m_triangleCellMap; }

    /**
     * @brief Computes the total number of settled particles across all triangular cells in the surface mesh.
     *
     * This method iterates through all stored triangle cells within the `SurfaceMesh` and
     * accumulates the particle count from each cell, returning the total number of settled particles.
     * The accumulation is performed using `std::accumulate` for optimized and functional-style iteration.
     *
     * @return size_t
     *         The total count of settled particles across all triangles in the surface mesh.
     *         If no particles have settled, the function returns `0`.
     *
     * @exception noexcept
     *            This function is marked `noexcept`, ensuring it does not throw exceptions.
     *
     * @complexity The time complexity of this method is **O(n)**, where `n` is the number of
     *             triangle cells in `m_triangleCellMap`, as it requires a single pass over all elements.
     *
     * @thread_safety This method is **not thread-safe** as it reads from `m_triangleCellMap`
     *                without synchronization. If multiple threads modify the surface mesh,
     *                external synchronization is required.
     *
     * @pre
     * - The `SurfaceMesh` instance should be fully initialized.
     * - The `m_triangleCellMap` should contain valid `TriangleCell` objects.
     *
     * @post
     * - The returned value is the sum of `count` attributes of all triangle cells.
     *
     * @note
     * - The function does not modify any member variables.
     * - If `m_triangleCellMap` is empty, the function returns `0`.
     *
     * **Example Usage:**
     * @code
     * SurfaceMesh surface("example.msh");
     * size_t totalParticles = surface.getTotalCountOfSettledParticles();
     * std::cout << "Total settled particles: " << totalParticles << std::endl;
     * @endcode
     *
     * **Implementation Details:**
     * - Uses `std::accumulate` with an initial sum of `0`.
     * - A lambda function extracts the `count` from each `TriangleCell` in `m_triangleCellMap`.
     */
    size_t getTotalCountOfSettledParticles() const noexcept;

    /**
     * @brief Gets the IDs of neighboring cells for a given triangle.
     * @param cellId ID of the target cell.
     * @return Vector of IDs of neighboring cells.
     * @throw GeometryCellIdNotFoundException If cellId does not exist.
     */
    std::vector<size_t> getNeighborCells(size_t cellId) const;
};

using SurfaceMesh_ref = SurfaceMesh &;
using SurfaceMesh_cref = SurfaceMesh const &;

#endif // !SURFACE_MESH_HPP
