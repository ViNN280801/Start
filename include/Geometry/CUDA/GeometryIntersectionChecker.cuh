#ifndef GEOMETRYINTERSECTIONCHECKER_CUH
#define GEOMETRYINTERSECTIONCHECKER_CUH

#ifdef USE_CUDA

#include "Geometry/AABBTreeDevice.cuh"
#include <cmath>

/**
 * @brief A class for checking geometric intersections on the GPU.
 *
 * This class provides methods to check intersections between rays and geometries,
 * such as Axis-Aligned Bounding Boxes (AABB) and triangles, optimized for CUDA.
 */
class GeometryIntersectionChecker
{
private:
    /**
     * @brief Swaps two float values.
     *
     * @param a Reference to the first value.
     * @param b Reference to the second value.
     */
    __device__ static void swap(float &a, float &b);

public:
    /**
     * @brief Checks if a ray intersects with an Axis-Aligned Bounding Box (AABB).
     *
     * @param[in] rayOrigin The origin of the ray in 3D space.
     * @param[in] rayDirInv The inverse of the ray's direction (1 / direction component) to optimize division operations.
     * @param[in] box The AABB defined by its minimum and maximum corners.
     * @return True if the ray intersects the AABB; otherwise, false.
     */
    __device__ static bool intersectRayAABB(Vec3Device_t const &rayOrigin, Vec3Device_t const &rayDirInv, AABBDevice_t const &box);

    /**
     * @brief Checks if a ray intersects with a triangle in 3D space.
     *
     * @param[in] rayOrigin The origin of the ray in 3D space.
     * @param[in] rayDir The direction of the ray in 3D space.
     * @param[in] tri The triangle defined by its three vertices in 3D space.
     * @param[out] t The distance from rayOrigin to the intersection point along rayDir.
     * @return True if the ray intersects the triangle; otherwise, false.
     */
    __device__ static bool intersectRayTriangle(Vec3Device_t const &rayOrigin, Vec3Device_t const &rayDir, TriangleDevice_t const &tri, float &t);
};

#endif // !USE_CUDA

#endif // !GEOMETRYINTERSECTIONCHECKER_CUH
