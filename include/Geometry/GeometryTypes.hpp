#ifndef GEOMETRY_TYPES_HPP
#define GEOMETRY_TYPES_HPP

#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_triangle_primitive_3.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Segment_3.h>
#include <CGAL/intersections.h>

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
    Triangle triangle; ///< Geometric representation of the triangle.
    double area;       ///< Precomputed area of the triangle (dS).
    size_t count;      ///< Counter of settled particles on this triangle.

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
    static double compute_area(Point p1, Point p2, Point p3) { return Kernel::Compute_area_3()(p1, p2, p3); }

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
    static double compute_area(Triangle triangle_) { return Kernel::Compute_area_3()(triangle_.vertex(0), triangle_.vertex(1), triangle_.vertex(2)); }
};

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

#endif // !GEOMETRY_TYPES_HPP
