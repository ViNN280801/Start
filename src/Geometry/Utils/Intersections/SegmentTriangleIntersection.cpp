#if __cplusplus <= 201703L
#include <variant>
#endif

#include "Geometry/Utils/Intersections/SegmentTriangleIntersection.hpp"

bool SegmentTriangleIntersection::isIntersectTriangle(Segment_cref segment, Triangle_cref triangle)
{
    return CGAL::do_intersect(segment, triangle);
}

std::optional<Point> SegmentTriangleIntersection::getIntersectionPoint(Segment_cref segment, Triangle_cref triangle)
{
    // Compute the intersection between the segment and the triangle.
    auto result{CGAL::intersection(segment, triangle)};
    if (!result)
        return std::nullopt;

// Use std::visit to handle the different possible types in the std::variant.
#if __cplusplus > 201703L
    return std::visit([](auto &&arg) -> std::optional<Point>
                      {
        using T = std::decay_t<decltype(arg)>;
        
        // If the intersection result is a Point, return it directly.
        if constexpr (std::is_same_v<T, Point>)
            return arg;
        // If the intersection result is a Segment (the entire segment lies in the triangle), return the source point.
        else if constexpr (std::is_same_v<T, Segment>)
            return arg.source();
        else
            return std::nullopt; }, *result);
#else
    if (const Point * p{std::get_if<Point>(boost::addressof(*result))})
        return *p;
    else if (const Segment * s{std::get_if<Segment>(boost::addressof(*result))})
        return s->source();
    return std::nullopt;
#endif
}
