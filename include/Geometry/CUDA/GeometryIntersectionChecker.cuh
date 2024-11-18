#ifndef GEOMETRYINTERSECTIONCHECKER_CUH
#define GEOMETRYINTERSECTIONCHECKER_CUH

#ifdef USE_CUDA

#include <cmath>

#include "Geometry/CUDA/AABBTreeDevice.cuh"

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
    __device__ static void swap(float &a, float &b)
    {
        float tmp = a;
        a = b;
        b = tmp;
    }

public:
    /**
     * @brief Checks if a ray intersects with an Axis-Aligned Bounding Box (AABB).
     *
     * @param[in] rayOrigin The origin of the ray in 3D space.
     * @param[in] rayDirInv The inverse of the ray's direction (1 / direction component) to optimize division operations.
     * @param[in] box The AABB defined by its minimum and maximum corners.
     * @return True if the ray intersects the AABB; otherwise, false.
     */
    __device__ static bool intersectRayAABB(Vec3Device_t const &rayOrigin, Vec3Device_t const &rayDirInv, AABBDevice_t const &box)
    {
        float tmin = (box.min.x - rayOrigin.x) * rayDirInv.x;
        float tmax = (box.max.x - rayOrigin.x) * rayDirInv.x;

        if (tmin > tmax)
            swap(tmin, tmax);

        float tymin = (box.min.y - rayOrigin.y) * rayDirInv.y;
        float tymax = (box.max.y - rayOrigin.y) * rayDirInv.y;

        if (tymin > tymax)
            swap(tymin, tymax);

        if ((tmin > tymax) || (tymin > tmax))
            return false;

        if (tymin > tmin)
            tmin = tymin;
        if (tymax < tmax)
            tmax = tymax;

        float tzmin = (box.min.z - rayOrigin.z) * rayDirInv.z;
        float tzmax = (box.max.z - rayOrigin.z) * rayDirInv.z;

        if (tzmin > tzmax)
            swap(tzmin, tzmax);

        if ((tmin > tzmax) || (tzmin > tmax))
            return false;

        return true;
    }

    /**
     * @brief Checks if a ray intersects with a triangle in 3D space.
     *
     * @param[in] rayOrigin The origin of the ray in 3D space.
     * @param[in] rayDir The direction of the ray in 3D space.
     * @param[in] tri The triangle defined by its three vertices in 3D space.
     * @param[out] t The distance from rayOrigin to the intersection point along rayDir.
     * @return True if the ray intersects the triangle; otherwise, false.
     */
    __device__ static bool intersectRayTriangle(Vec3Device_t const &rayOrigin, Vec3Device_t const &rayDir, TriangleDevice_t const &tri, float &t)
    {
        const float EPSILON = 1e-6f;

        Vec3Device_t edge1, edge2, h, s, q;
        float a, f, u, v;

        edge1.x = tri.v1.x - tri.v0.x;
        edge1.y = tri.v1.y - tri.v0.y;
        edge1.z = tri.v1.z - tri.v0.z;

        edge2.x = tri.v2.x - tri.v0.x;
        edge2.y = tri.v2.y - tri.v0.y;
        edge2.z = tri.v2.z - tri.v0.z;

        h.x = rayDir.y * edge2.z - rayDir.z * edge2.y;
        h.y = rayDir.z * edge2.x - rayDir.x * edge2.z;
        h.z = rayDir.x * edge2.y - rayDir.y * edge2.x;

        a = edge1.x * h.x + edge1.y * h.y + edge1.z * h.z;

        if (fabsf(a) < EPSILON)
            return false;

        f = 1.0f / a;

        s.x = rayOrigin.x - tri.v0.x;
        s.y = rayOrigin.y - tri.v0.y;
        s.z = rayOrigin.z - tri.v0.z;

        u = f * (s.x * h.x + s.y * h.y + s.z * h.z);
        if (u < 0.0f || u > 1.0f)
            return false;

        q.x = s.y * edge1.z - s.z * edge1.y;
        q.y = s.z * edge1.x - s.x * edge1.z;
        q.z = s.x * edge1.y - s.y * edge1.x;

        v = f * (rayDir.x * q.x + rayDir.y * q.y + rayDir.z * q.z);
        if (v < 0.0f || u + v > 1.0f)
            return false;

        t = f * (edge2.x * q.x + edge2.y * q.y + edge2.z * q.z);

        return t > EPSILON;
    }
};

#endif // !USE_CUDA

#endif // !GEOMETRYINTERSECTIONCHECKER_CUH
