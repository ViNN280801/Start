#ifdef USE_OMP
#include <omp.h>
#endif

#include "Geometry/Utils/Overlaps/TriangleOverlapCalculator.hpp"

double TriangleOverlapCalculator::calculateOverlapRatio(triangle3d_projected_cref tri, double x_min, double x_max)
{
    // 1. Create 2D triangle projection
    Polygon_2 triangle_poly{_createProjectedTriangle(tri)};

    // 2. Create band polygon
    Polygon_2 band{_createBandPolygon(tri, x_min, x_max)};

    // 3. Calculate intersection
#if __cplusplus <= 201703L
    std::list<Polygon_with_holes_2> ipolys;
#else
    std::vector<Polygon_with_holes_2> ipolys;
#endif
    CGAL::intersection(triangle_poly, band, std::back_inserter(ipolys));

    // 4. Calculate areas
    double area_triangle{std::abs(triangle_poly.area())},
        area_intersection{_calculateIntersectionArea(ipolys)};

    // 5. If triangle area is zero, return 0.0, otherwise return the ratio of intersection area to triangle area.
    return (area_triangle > 0) ? (area_intersection / area_triangle) : 0.0;
}

Polygon_2 TriangleOverlapCalculator::_createProjectedTriangle(triangle3d_projected_cref tri)
{
    Polygon_2 triangle_poly;
    triangle_poly.push_back(Point_2(tri.vertex(0).x(), tri.vertex(0).y()));
    triangle_poly.push_back(Point_2(tri.vertex(1).x(), tri.vertex(1).y()));
    triangle_poly.push_back(Point_2(tri.vertex(2).x(), tri.vertex(2).y()));
    return triangle_poly;
}

Polygon_2 TriangleOverlapCalculator::_createBandPolygon(triangle3d_projected_cref tri, double x_min, double x_max)
{
    // 1. Calculate min and max y coordinates of the triangle to create a band polygon.
    // Band is a polygon that is parallel to the X axis and passes through the triangle.
    double y_min{std::min({tri.vertex(0).y(), tri.vertex(1).y(), tri.vertex(2).y()}) - 1.0},
        y_max{std::max({tri.vertex(0).y(), tri.vertex(1).y(), tri.vertex(2).y()}) + 1.0};

    // 2. Create a band polygon by connecting the points:
    // (x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max).
    Polygon_2 band;
    band.push_back(Point_2(x_min, y_min));
    band.push_back(Point_2(x_max, y_min));
    band.push_back(Point_2(x_max, y_max));
    band.push_back(Point_2(x_min, y_max));
    return band;
}

double TriangleOverlapCalculator::_calculateIntersectionArea(intersection_polys_cref intersection_polys)
{
    double area_intersection{};

    // 1. Convert list to vector for OpenMP compatibility.
    std::vector<Polygon_with_holes_2> polys(intersection_polys.begin(), intersection_polys.end());

    // 2. Calculate area of the intersection polygons.
#ifdef USE_OMP
#pragma omp parallel for reduction(+ : area_intersection) num_threads(omp_get_max_threads())
#endif
    for (size_t i = 0; i < polys.size(); ++i)
    {
        // 2.1. Calculate area of the outer boundary. PWH - Polygon with holes.
        double pwh_area{std::abs(polys[i].outer_boundary().area())};

        // 2.2. Subtract holes area.
        for (auto hit = polys[i].holes_begin(); hit != polys[i].holes_end(); ++hit)
            pwh_area -= std::abs(hit->area());

        // 2.3. Add to the total area.
        area_intersection += pwh_area;
    }

    return area_intersection;
}
