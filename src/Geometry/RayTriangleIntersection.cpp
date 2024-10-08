#include "Geometry/RayTriangleIntersection.hpp"

bool RayTriangleIntersection::isIntersectTriangleImpl(Ray const &ray, Triangle const &triangle)
{
    return CGAL::do_intersect(ray, triangle);
}

std::optional<Point> RayTriangleIntersection::getIntersectionPointImpl(Ray const &ray, Triangle const &triangle)
{
    // Compute the intersection between the ray and the triangle.
    auto result{CGAL::intersection(ray, triangle)};
    if (!result)
        return std::nullopt;

    // Use std::visit to handle the different possible types in the std::variant.
    return std::visit([](auto &&arg) -> std::optional<Point>
                      {
        using T = std::decay_t<decltype(arg)>;
        
        // If the intersection result is a Point, return it directly.
        if constexpr (std::is_same_v<T, Point>)
            return arg;
        // If the intersection result is a Ray (the entire ray lies in the triangle), return the source point.
        else if constexpr (std::is_same_v<T, Ray>)
            return arg.source();
        else
            return std::nullopt; }, *result);
}
