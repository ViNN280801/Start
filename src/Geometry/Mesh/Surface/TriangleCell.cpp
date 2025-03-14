#include "Geometry/Mesh/Surface/TriangleCell.hpp"

TriangleCell::TriangleCell(Triangle_cref tri, double area_, size_t count_)
    : triangle(tri), area(area_), count(count_) {}

Point TriangleCell::compute_centroid(Point_cref p1, Point_cref p2, Point_cref p3)
{
    return CGAL::centroid(p1, p2, p3);
}

Point TriangleCell::compute_centroid(Triangle_cref triangle_)
{
    return CGAL::centroid(triangle_.vertex(0), triangle_.vertex(1), triangle_.vertex(3));
}

double TriangleCell::compute_area(Point_cref p1, Point_cref p2, Point_cref p3)
{
    return Kernel::Compute_area_3()(p1, p2, p3);
}

double TriangleCell::compute_area(Triangle_cref triangle_)
{
    return Kernel::Compute_area_3()(triangle_.vertex(0), triangle_.vertex(1), triangle_.vertex(2));
}
