#include <gtest/gtest.h>

#include "Geometry/Basics/BaseTypes.hpp"
#include "Geometry/Utils/Intersections/SegmentTriangleIntersection.hpp"

TEST(SegmentTriangleIntersectionTest, isIntersectTriangle)
{
    EXPECT_TRUE(SegmentTriangleIntersection::isIntersectTriangle(
        Segment(Point(0, 0, 0), Point(0, 0, 1)),
        Triangle(Point(-1, 1, 0.5), Point(1, 1, 0.5), Point(0, -1, 0.5))));
    EXPECT_FALSE(SegmentTriangleIntersection::isIntersectTriangle(
        Segment(Point(1, 0, 0), Point(1, 0, 1)),
        Triangle(Point(-1, 1, 0.5), Point(1, 1, 0.5), Point(0, -1, 0.5))));
    EXPECT_FALSE(SegmentTriangleIntersection::isIntersectTriangle(
        Segment(Point(-1, 0, 0), Point(-1, 0, 1)),
        Triangle(Point(-1, 1, 0.5), Point(1, 1, 0.5), Point(0, -1, 0.5))));

    EXPECT_TRUE(SegmentTriangleIntersection::isIntersectTriangle(
        Segment(Point(0, 0, 0), Point(1, 0, 0)),
        Triangle(Point(0.5, 1, 1), Point(0.5, -1, 1), Point(0.5, 0, -1))));
    EXPECT_FALSE(SegmentTriangleIntersection::isIntersectTriangle(
        Segment(Point(-1, 0, 0), Point(1, -1, 0)),
        Triangle(Point(0.5, 1, 1), Point(0.5, -1, 1), Point(0.5, 0, -1))));
    EXPECT_FALSE(SegmentTriangleIntersection::isIntersectTriangle(
        Segment(Point(-1, 0, 0), Point(1, 1, 0)),
        Triangle(Point(0.5, 1, 1), Point(0.5, -1, 1), Point(0.5, 0, -1))));
}

TEST(SegmentTriangleIntersectionTest, SegmentLocating)
{
    auto coordinate = SegmentTriangleIntersection::getIntersectionPoint(
        Segment(Point(0, 0, 0), Point(0, 0, 1)),
        Triangle(Point(-1, 1, 0.5), Point(1, 1, 0.5), Point(0, -1, 0.5)));
    EXPECT_NEAR(coordinate->x(), 0, 0.00001);
    EXPECT_NEAR(coordinate->y(), 0, 0.00001);
    EXPECT_NEAR(coordinate->z(), 0.5, 0.00001);

    auto coordinate_s = SegmentTriangleIntersection::getIntersectionPoint(
        Segment(Point(0, 0, 0), Point(1, 0, 0)),
        Triangle(Point(0.5, 1, 1), Point(0.5, -1, 1), Point(0.5, 0, -1)));
    EXPECT_NEAR(coordinate_s->x(), 0.5, 0.00001);
    EXPECT_NEAR(coordinate_s->y(), 0, 0.00001);
    EXPECT_NEAR(coordinate_s->z(), 0, 0.00001);
}
