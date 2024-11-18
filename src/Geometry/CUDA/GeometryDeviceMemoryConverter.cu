#ifdef USE_CUDA
#include "GeometryDeviceMemoryConverter.cuh"

/**
 * @brief Converts a host-side Triangle to its device representation.
 */
TriangleDevice_t GeometryDeviceMemoryConverter::TriangleToDevice(Triangle const &triangle)
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

/**
 * @brief Converts a device-side Triangle back to its host representation.
 */
Triangle GeometryDeviceMemoryConverter::TriangleToHost(TriangleDevice_t const &triangleDevice)
{
    return Triangle(
        Point(triangleDevice.v0.x, triangleDevice.v0.y, triangleDevice.v0.z),
        Point(triangleDevice.v1.x, triangleDevice.v1.y, triangleDevice.v1.z),
        Point(triangleDevice.v2.x, triangleDevice.v2.y, triangleDevice.v2.z));
}

/**
 * @brief Converts a host-side Vec3 to its device representation.
 */
Vec3Device_t GeometryDeviceMemoryConverter::Vec3ToDevice(MathVector<double> const &vec)
{
    return Vec3Device_t{vec.x, vec.y, vec.z};
}

/**
 * @brief Converts a device-side Vec3 back to its host representation.
 */
MathVector<double> GeometryDeviceMemoryConverter::Vec3ToHost(Vec3Device_t const &vecDevice)
{
    return MathVector<double>{vecDevice.x, vecDevice.y, vecDevice.z};
}

#endif // !USE_CUDA
