#ifndef GEOMETRYDEVICEMEMORYCONVERTER_CUH
#define GEOMETRYDEVICEMEMORYCONVERTER_CUH

#ifdef USE_CUDA
#include "Geometry/GeometryTypes.cuh"
#include "Geometry/MathVector.hpp"

/**
 * @brief Class providing utility methods for converting geometry data between host and device representations.
 *
 * This class offers type-specific methods to convert objects such as triangles and vectors
 * between host and device memory formats. It ensures compatibility with CUDA-based computations.
 */
class GeometryDeviceMemoryConverter
{
public:
    /**
     * @brief Converts a host-side Triangle to its device representation.
     * @param triangle The host-side Triangle.
     * @return A TriangleDevice_t object representing the device-side Triangle.
     */
    static TriangleDevice_t TriangleToDevice(Triangle const &triangle);

    /**
     * @brief Converts a device-side Triangle back to its host representation.
     * @param triangleDevice The device-side TriangleDevice_t.
     * @return A Triangle object representing the host-side Triangle.
     */
    static Triangle TriangleToHost(TriangleDevice_t const &triangleDevice);

    /**
     * @brief Converts a host-side MathVector<double> to its device representation.
     * @param vec The host-side MathVector<double> object.
     * @return A Vec3Device_t object representing the device-side vector.
     */
    static Vec3Device_t Vec3ToDevice(MathVector<double> const &vec);

    /**
     * @brief Converts a device-side MathVector<double> back to its host representation.
     * @param vecDevice The device-side Vec3Device_t object.
     * @return A MathVector<double> object representing the host-side vector.
     */
    static MathVector<double> Vec3ToHost(Vec3Device_t const &vecDevice);
};

#endif // !USE_CUDA

#endif // !GEOMETRYDEVICEMEMORYCONVERTER_CUH
