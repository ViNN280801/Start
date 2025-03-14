#ifndef SEGMENT_TRIANGLE_INTERSECTION_HPP
#define SEGMENT_TRIANGLE_INTERSECTION_HPP

#include "Geometry/Basics/BaseTypes.hpp"

class SegmentTriangleIntersection
{
public:
    /**
     * @brief Checker for segment-triangle intersection.
     * @param segment Segment object.
     * @param triangle Triangle object.
     * @return `true` if segment intersects the triangle, otherwise `false`.
     */
    [[nodiscard("Ignoring the intersection test result can lead to incorrect geometric or physical computations.")]]
    static bool isIntersectTriangle(Segment_cref segment, Triangle_cref triangle);

    /**
     * @brief Computes the intersection point of the segment with a given triangle.
     * @param segment Segment object.
     * @param triangle Triangle object.
     * @return A `std::optional<Point>` containing the intersection point if it exists;
     * otherwise, `std::nullopt`.
     */
    [[nodiscard("Ignoring the intersection point may lead to incorrect behavior in applications relying on accurate geometric calculations.")]]
    static std::optional<Point> getIntersectionPoint(Segment_cref segment, Triangle_cref triangle);
};

#endif // !SEGMENT_TRIANGLE_INTERSECTION_HPP
