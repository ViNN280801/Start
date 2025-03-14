#ifndef TRIANGLE_CELL_HPP
#define TRIANGLE_CELL_HPP

#include "Geometry/Basics/BaseTypes.hpp"
#include "Geometry/Mesh/Surface/AABBTree/AABBTree.hpp"
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

    TriangleCell() = default;
    TriangleCell(Triangle_cref tri, double area_, size_t count_);

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
    static Point compute_centroid(Point_cref p1, Point_cref p2, Point_cref p3);

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
    static Point compute_centroid(Triangle_cref triangle_);

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
    static double compute_area(Point_cref p1, Point_cref p2, Point_cref p3);

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
    static double compute_area(Triangle_cref triangle_);
};
using TriangleCell_ref = TriangleCell &;
using TriangleCell_cref = TriangleCell const &;

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
using TriangleCellMap_ref = TriangleCellMap &;
using TriangleCellMap_cref = TriangleCellMap const &;

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
 */
using TriangleCellCentersMap = std::unordered_map<size_t, Point>;
using TriangleCellCentersMap_ref = TriangleCellCentersMap &;
using TriangleCellCentersMap_cref = TriangleCellCentersMap const &;

#endif // !TRIANGLE_CELL_HPP
