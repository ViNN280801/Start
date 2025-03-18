#ifndef TRIANGLE_OVERLAP_CALCULATOR_HPP
#define TRIANGLE_OVERLAP_CALCULATOR_HPP

#include <CGAL/Boolean_set_operations_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Polygon_with_holes_2.h>

#if __cplusplus <= 201703L
#include <list>
#else
#include <span>
#endif

using Kernel2 = CGAL::Exact_predicates_inexact_constructions_kernel; ///< CGAL kernel for 2D geometry.
using Point_2 = Kernel2::Point_2;                                    ///< 2D point type.
using Polygon_2 = CGAL::Polygon_2<Kernel2>;                          ///< 2D polygon type.
using Polygon_with_holes_2 = CGAL::Polygon_with_holes_2<Kernel2>;    ///< 2D polygon with holes type.

/**
 * @class TriangleOverlapCalculator
 * @brief Calculates the overlap ratio between a 3D triangle and a vertical band in 2D space.
 *
 * This class provides functionality to calculate the overlap ratio between a 3D triangle
 * and a vertical band by projecting the triangle onto the XY plane and computing the
 * intersection area with the band.
 */
class TriangleOverlapCalculator
{
private:
#if __cplusplus <= 201703L
    using intersection_polys = std::list<Polygon_with_holes_2>; ///< List of polygons with holes representing the intersection.
    using intersection_polys_ref = intersection_polys &;        ///< Reference to the list of intersection polygons.
    using intersection_polys_cref = intersection_polys const &; ///< Constant reference to the list of intersection polygons.
#else
    using intersection_polys = std::span<Polygon_with_holes_2>;            ///< Span of polygons with holes representing the intersection.
    using intersection_polys_ref = intersection_polys &;                   ///< Reference to the span of intersection polygons.
    using intersection_polys_cref = std::span<Polygon_with_holes_2 const>; ///< Constant reference to the span of intersection polygons.
#endif

    /**
     * @brief 3D triangle, but projected onto the XY plane.
     * This allows using 2D kernel for optimizations.
     */
    using triangle3d_projected = CGAL::Triangle_3<Kernel2>;
    using triangle3d_projected_ref = triangle3d_projected &;
    using triangle3d_projected_cref = triangle3d_projected const &;

    /**
     * @brief Creates a 2D polygon by projecting a 3D triangle onto the XY plane.
     * @param tri The 3D triangle to project.
     * @return Polygon_2 representing the projected triangle.
     */
    static Polygon_2 _createProjectedTriangle(triangle3d_projected_cref tri);

    /**
     * @brief Creates a vertical band polygon based on the triangle's Y extent.
     * @param tri The 3D triangle used to determine the Y extent.
     * @param x_min The minimum X coordinate of the band.
     * @param x_max The maximum X coordinate of the band.
     * @return Polygon_2 representing the vertical band.
     */
    static Polygon_2 _createBandPolygon(triangle3d_projected_cref tri, double x_min, double x_max);

    /**
     * @brief Calculates the total area of intersection polygons.
     * @param intersection_polys List of polygons with holes representing the intersection.
     * @return The total area of the intersection.
     */
    static double _calculateIntersectionArea(intersection_polys_cref intersection_polys);

public:
    /**
     * @brief Calculates the overlap ratio between a 3D triangle and a vertical band.
     * @param tri The 3D triangle to check for overlap.
     * @param x_min The minimum X coordinate of the band.
     * @param x_max The maximum X coordinate of the band.
     * @return The ratio of intersection area to triangle area (0.0 to 1.0).
     *
     * @algorithm
     * 1. Project the 3D triangle onto the XY plane.
     * 2. Create a vertical band polygon based on the triangle's Y extent.
     * 3. Calculate the intersection between the projected triangle and band.
     * 4. Compute the area of the intersection.
     * 5. Return the ratio of intersection area to triangle area.
     *
     * @exception_safety Strong guarantee - if an exception occurs, the program state remains unchanged.
     */
    static double calculateOverlapRatio(triangle3d_projected_cref tri, double x_min, double x_max);
};

#endif // !TRIANGLE_OVERLAP_CALCULATOR_HPP
