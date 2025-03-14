#include "Geometry/Basics/Edge.hpp"

edge_t::edge_t(Point_cref a, Point_cref b) : p1(a < b ? a : b), p2(a < b ? b : a) {}

size_t edge_hash_t::operator()(edge_t_cref e) const noexcept
{
    auto hash1{CGAL::hash_value(e.p1)};
    auto hash2{CGAL::hash_value(e.p2)};
    return hash1 ^ (hash2 << 1);
}

edge_t_vector getTriangleEdges(Triangle_cref triangle) noexcept
{
    return {
        edge_t(triangle.vertex(0), triangle.vertex(1)),
        edge_t(triangle.vertex(1), triangle.vertex(2)),
        edge_t(triangle.vertex(2), triangle.vertex(0))};
}

bool operator==(edge_t_cref a, edge_t_cref b) noexcept
{
    return (a.p1 == b.p1 && a.p2 == b.p2) || (a.p1 == b.p2 && a.p2 == b.p1);
}

