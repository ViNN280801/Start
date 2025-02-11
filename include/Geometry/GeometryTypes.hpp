#ifndef GEOMETRY_TYPES_HPP
#define GEOMETRY_TYPES_HPP

#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_triangle_primitive_3.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Segment_3.h>
#include <CGAL/intersections.h>

#include "Utilities/Utilities.hpp"

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel; ///< Kernel for exact predicates and inexact constructions.
using Point = Kernel::Point_3;                                      ///< 3D point type.
using Ray = Kernel::Segment_3;                                      ///< Finite ray (line segment).
using Triangle = Kernel::Triangle_3;                                ///< 3D triangle type.
using Tetrahedron = Kernel::Tetrahedron_3;                          ///< 3D tetrahedron type.

/**
 * @brief Represents the parameters of a triangle cell in a mesh (for surface modeling).
 *
 * The `TriangleCell` structure is used to store geometric and simulation-related properties
 * of a single triangle in a surface mesh. Each triangle is uniquely identified by its ID
 * (as a key in `TriangleCellMap`) and contains:
 * - The geometric representation of the triangle.
 * - The precomputed area of the triangle.
 * - A counter tracking the number of settled particles on this triangle.
 */
struct TriangleCell
{
    Triangle triangle;                  ///< Geometric representation of the triangle.
    double area;                        ///< Precomputed area of the triangle (dS).
    size_t count;                       ///< Counter of settled particles on this triangle.
    std::vector<size_t> neighbor_ids{}; ///< IDs of neighboring cells.

    /**
     * @brief Computes the geometric centroid of a triangle from its vertices.
     *
     * This static method calculates the centroid (geometric center) of a triangle in 3D space
     * using the coordinates of its three vertices. The centroid is computed as the average
     * of the three vertices' coordinates:
     * \f[
     * C_x = \frac{x_1 + x_2 + x_3}{3}, x1+x2+x3/3
     * C_y = \frac{y_1 + y_2 + y_3}{3}, y1+y2+y3/3
     * C_z = \frac{z_1 + z_2 + z_3}{3}, z1+z2+z3/3
     * \f]
     *
     * @param p1 First vertex of the triangle (Kernel::Point_3)
     * @param p2 Second vertex of the triangle (Kernel::Point_3)
     * @param p3 Third vertex of the triangle (Kernel::Point_3)
     * @return Point Centroid coordinates as a CGAL::Point_3
     *
     * @note For degenerate triangles (all points collinear), the centroid will lie on the line
     *       segment connecting the points. The calculation remains mathematically valid even
     *       for zero-area triangles.
     */
    static Point compute_centroid(Point const &p1, Point const &p2, Point const &p3) { return CGAL::centroid(p1, p2, p3); }

    /**
     * @brief Computes the geometric centroid of a triangle from its vertices.
     *
     * This static method calculates the centroid (geometric center) of a triangle in 3D space
     * using the coordinates of its three vertices. The centroid is computed as the average
     * of the three vertices' coordinates:
     * \f[
     * C_x = \frac{x_1 + x_2 + x_3}{3}, x1+x2+x3/3
     * C_y = \frac{y_1 + y_2 + y_3}{3}, y1+y2+y3/3
     * C_z = \frac{z_1 + z_2 + z_3}{3}, z1+z2+z3/3
     * \f]
     *
     * @param triangle_ The triangle whose area is to be computed.
     * @return Point Centroid coordinates as a CGAL::Point_3
     *
     * @note For degenerate triangles (all points collinear), the centroid will lie on the line
     *       segment connecting the points. The calculation remains mathematically valid even
     *       for zero-area triangles.
     */
    static Point compute_centroid(Triangle const &triangle_) { return CGAL::centroid(triangle_.vertex(0), triangle_.vertex(1), triangle_.vertex(3)); }

    /**
     * @brief Computes the area of a triangle given its three vertices.
     *
     * This static method calculates the area of a triangle in 3D space using the coordinates
     * of its three vertices. The computation leverages the CGAL kernel's `Compute_area_3` function
     * to ensure precision.
     *
     * @param p1 The first vertex of the triangle.
     * @param p2 The second vertex of the triangle.
     * @param p3 The third vertex of the triangle.
     * @return double The computed area of the triangle.
     *
     * @note The method assumes that the points form a valid triangle. If the points are collinear
     *       or degenerate, the area will be zero.
     */
    static double compute_area(Point const &p1, Point const &p2, Point const &p3) { return Kernel::Compute_area_3()(p1, p2, p3); }

    /**
     * @brief Computes the area of a triangle represented as a `Triangle` object.
     *
     * This static method calculates the area of a triangle in 3D space using a `Triangle` object
     * as input. The method extracts the vertices of the triangle and uses the CGAL kernel's
     * `Compute_area_3` function for the computation.
     *
     * @param triangle_ The triangle whose area is to be computed.
     * @return double The computed area of the triangle.
     *
     * @note If the triangle is degenerate (e.g., its vertices are collinear), the area will be zero.
     */
    static double compute_area(Triangle const &triangle_) { return Kernel::Compute_area_3()(triangle_.vertex(0), triangle_.vertex(1), triangle_.vertex(2)); }
};

/**
 * @brief Edge struct.
 * @details This struct is used to store the edges of the triangles.
 */
struct Edge
{
    Point p1, p2;
    Edge(const Point &a, const Point &b) : p1(a < b ? a : b), p2(a < b ? b : a) {}
};

/**
 * @brief getTriangleEdges function.
 * @details This function is used to get the edges of the triangles.
 * @param triangle The triangle to get the edges of.
 * @return std::vector<Edge> The edges of the triangle.
 */
std::vector<Edge> getTriangleEdges(Triangle const &triangle);

/**
 * @brief EdgeHash struct.
 * @details This struct is used to hash the edges of the triangles.
 */
struct EdgeHash 
{
    size_t operator()(const Edge &e) const 
    {
        auto hash1{CGAL::hash_value(e.p1)};
        auto hash2{CGAL::hash_value(e.p2)};
        return hash1 ^ (hash2 << 1);
    }
};

/**
 * @brief operator== for Edge struct.
 * @details This operator is used to compare the edges of the triangles.
 */
bool operator==(const Edge &a, const Edge &b) noexcept;

/**
 * @brief A map for storing and accessing triangle cells by their unique IDs.
 *
 * The `TriangleCellMap` provides an efficient hash-based container for triangle cells.
 * Each entry consists of:
 * - Key: A unique identifier for the triangle (`size_t`).
 * - Value: A `TriangleCell` structure containing geometric and simulation data.
 *
 * This map is designed for efficient lookup of triangle cells in large-scale surface meshes.
 */
using TriangleCellMap = std::unordered_map<size_t, TriangleCell>;

/**
 * @brief A map for storing and accessing the geometric centers of triangle cells by their unique IDs.
 *
 * The `TriangleCellCentersMap` provides an efficient hash-based container associating triangle identifiers
 * with their corresponding geometric centers. Each entry consists of:
 * - Key: A unique identifier for the triangle (`size_t`), consistent with other mesh-related maps.
 * - Value: A 3-element array containing the XYZ coordinates of the triangle's center.
 *
 * The center is typically calculated as the centroid (geometric center) of the triangle's three vertices,
 * computed using the formula: \f$( \frac{x_1+x_2+x_3}{3}, \frac{y_1+y_2+y_3}{3}, \frac{z_1+z_2+z_3}{3} )\f$.
 *                              [(x1+x2+x3)/3; (y1+y2+y3)/3; (z1+z2+z3)/3]
 *
 * @note The coordinates are stored as `double` values in a `std::array<double, 3>` for memory efficiency
 *       and cache-friendly access patterns. The order of coordinates is [X, Y, Z].
 *
 * Example usage:
 * @code
 * TriangleCellCentersMap centers;
 * size_t triangle_id = 42;
 * auto& center = centers[triangle_id]; // Access centroid of triangle 42
 * double x = center[0], y = center[1], z = center[2];
 * @endcode
 */
using TriangleCellCentersMap = std::unordered_map<size_t, std::array<double, 3ul>>;

/**
 * @brief Represents the parameters of a tetrahedron cell in a 3D mesh.
 *
 * The `TetrahedronCell` structure is used to store the geometric representation of a
 * tetrahedron in a volumetric mesh. Each tetrahedron is uniquely identified by its ID
 * (as a key in `TetrahedronCellMap`) and contains:
 * - The geometric representation of the tetrahedron.
 */
struct TetrahedronCell
{
    Tetrahedron tetrahedron; ///< Geometric representation of the tetrahedron.
};

/**
 * @brief A map for storing and accessing tetrahedron cells by their unique IDs.
 *
 * The `TetrahedronCellMap` provides an efficient hash-based container for tetrahedron cells.
 * Each entry consists of:
 * - Key: A unique identifier for the tetrahedron (`size_t`).
 * - Value: A `TetrahedronCell` structure containing geometric data.
 *
 * This map is suitable for managing volumetric meshes with efficient access patterns.
 */
using TetrahedronCellMap = std::unordered_map<size_t, TetrahedronCell>;

using TriangleVector = std::vector<Triangle>;                                               ///< Vector of triangles.
using TriangleVectorConstIter = TriangleVector::const_iterator;                             ///< Constant iterator for a vector of triangles.
using TrianglePrimitive = CGAL::AABB_triangle_primitive_3<Kernel, TriangleVectorConstIter>; ///< Primitive for representing triangles in an AABB tree for efficient spatial queries.
using TriangleTraits = CGAL::AABB_traits_3<Kernel, TrianglePrimitive>;                      ///< Traits class defining the properties and operations of triangle primitives for use in an AABB tree.
using AABB_Tree_Triangle = CGAL::AABB_tree<TriangleTraits>;                                 ///< Axis-Aligned Bounding Box (AABB) tree for accelerating spatial queries (e.g., intersections, nearest neighbors) on triangles.

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
    AABB_Tree_Triangle m_aabbTree;     ///< AABB tree for efficient spatial queries.

    /**
     * @brief Fills the triangle vector with valid, non-degenerate triangles.
     *
     * This method iterates over `m_triangleCellMap`, extracting valid triangles and
     * filtering out **degenerate** ones (triangles with zero area).
     * - Degenerate triangles are **logged** and **skipped**.
     * - If all triangles are degenerate, an exception is thrown.
     *
     * @throws std::runtime_error If no valid triangles are found after filtering.
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
     * @throws std::runtime_error If `m_triangles` is empty or if the AABB tree cannot be built.
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
     *
     * @throws std::runtime_error If any step in the initialization fails.
     */
    void _initialize();

public:
    /**
     * @brief Constructs a SurfaceMesh from a given mesh file.
     *
     * This constructor reads a `.msh` file and extracts all triangular cells present in the mesh.
     * - Calls `Mesh::getMeshParams()` to retrieve mesh data.
     * - Initializes the **AABB tree** for spatial queries.
     *
     * @param meshFilename The path to the Gmsh `.msh` file.
     * @throw std::runtime_error If the file is missing, invalid, or contains no valid triangles.
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
     * @throw std::invalid_argument If `triangleCells` is empty.
     */
    explicit SurfaceMesh(TriangleCellMap const &triangleCells);

    /**
     * @brief Constructs a SurfaceMesh from a specific physical group in a mesh file.
     *
     * This constructor extracts **only** the triangles associated with a specified **physical group**.
     * - Uses `GmshUtils::getCellsByPhysicalGroupName()` to retrieve triangle data.
     * - Builds the **AABB tree** after loading.
     *
     * @param meshFilename The path to the Gmsh `.msh` file.
     * @param physicalGroupName The name of the physical group containing the desired triangles.
     * @throw std::runtime_error If the physical group does not exist or contains no valid triangles.
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
    AABB_Tree_Triangle const &getAABBTree() const { return m_aabbTree; }

    /**
     * @brief Provides read-only access to the stored triangle vector.
     * @return const TriangleVector& Reference to the vector of non-degenerate triangles.
     *
     * **Note:**
     * - This vector only contains **valid (non-degenerate)** triangles.
     * - It is primarily used for **rendering** or **direct triangle iteration**.
     */
    constexpr TriangleVector const &getTriangles() const { return m_triangles; }

    /**
     * @brief Provides read-only access to the triangle cell map.
     * @return const TriangleCellMap& Reference to the triangle cell map.
     *
     * The **triangle cell map** allows direct access to:
     * - Triangle geometry.
     * - Triangle area.
     * - Particle counts (if applicable in a simulation).
     */
    constexpr TriangleCellMap const &getTriangleCellMap() const { return m_triangleCellMap; }

    /**
     * @brief Provides read/write access to the triangle cell map.
     * @return TriangleCellMap& Reference to the triangle cell map.
     *
     * Allows modification of the triangle data, but changes may require **rebuilding the AABB tree**.
     * - Modifications to triangle geometry should be followed by `_constructAABB()`.
     */
    constexpr TriangleCellMap &getTriangleCellMap() { return m_triangleCellMap; }

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
     * @throw std::out_of_range If cellId does not exist.
     */
    std::vector<size_t> getNeighborCells(size_t cellId) const;
};

#endif // !GEOMETRY_TYPES_HPP
