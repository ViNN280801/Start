#ifndef GEOMETRYDEVICEMEMORYCONVERTER_CUH
#define GEOMETRYDEVICEMEMORYCONVERTER_CUH

#ifdef USE_CUDA
#include "Geometry/GeometryTypes.hpp"

TriangleDevice_t TriangleToDevice(Triangle const &triangle)
{
    TriangleDevice_t t;
    auto v0 = triangle.vertex(0);
    auto v1 = triangle.vertex(1);
    auto v2 = triangle.vertex(2);

    t.v0 = {v0.x(), v0.y(), v0.z()};
    t.v1 = {v1.x(), v1.y(), v1.z()};
    t.v2 = {v2.x(), v2.y(), v2.z()};

    return t;
}
#endif

#endif // !GEOMETRYDEVICEMEMORYCONVERTER_CUH
