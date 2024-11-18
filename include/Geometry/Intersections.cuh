#ifndef INTERSECTIONS_CUH
#define INTERSECTIONS_CUH

#ifdef USE_CUDA

#include "Geometry/AABBTreeDevice.cuh"

__device__ void myswap(float &a, float &b)
{
    float tmp = a;
    a = b;
    b = tmp;
}

/**
 * @brief Checks if a ray intersects with an Axis-Aligned Bounding Box (AABB).
 *
 * The function tests the intersection of a ray defined by its origin and inverse direction
 * with an AABB defined by its minimum and maximum corners. It performs the intersection
 * check based on the slab method, which involves calculating intersection points along each axis.
 *
 * @param[in] rayOrigin The origin of the ray in 3D space.
 * @param[in] rayDirInv The inverse of the ray's direction (1 / direction component) to optimize division operations.
 * @param[in] box The AABB defined by its minimum and maximum corners.
 * @return True if the ray intersects the AABB; otherwise, false.
 */
__device__ bool intersectRayAABB(const Vec3Device_t &rayOrigin, const Vec3Device_t &rayDirInv, const AABBDevice_t &box)
{
    // Compute intersection parameters for the X axis
    float tmin = (box.min.x - rayOrigin.x) * rayDirInv.x;
    float tmax = (box.max.x - rayOrigin.x) * rayDirInv.x;

    // Swap if tmin is greater than tmax
    if (tmin > tmax)
        myswap(tmin, tmax);

    // Compute intersection parameters for the Y axis
    float tymin = (box.min.y - rayOrigin.y) * rayDirInv.y;
    float tymax = (box.max.y - rayOrigin.y) * rayDirInv.y;

    // Swap if tymin is greater than tymax
    if (tymin > tymax)
        myswap(tymin, tymax);

    // Check if intervals on X and Y overlap
    if ((tmin > tymax) || (tymin > tmax))
        return false;

    // Update tmin and tmax based on Y interval
    if (tymin > tmin)
        tmin = tymin;
    if (tymax < tmax)
        tmax = tymax;

    // Compute intersection parameters for the Z axis
    float tzmin = (box.min.z - rayOrigin.z) * rayDirInv.z;
    float tzmax = (box.max.z - rayOrigin.z) * rayDirInv.z;

    // Swap if tzmin is greater than tzmax
    if (tzmin > tzmax)
        myswap(tzmin, tzmax);

    // Check if intervals on X, Y, and Z overlap
    if ((tmin > tzmax) || (tzmin > tmax))
        return false;

    // Ray intersects with the AABB
    return true;
}

/**
 * @brief Checks if a ray intersects with a triangle in 3D space.
 *
 * This function uses the Möller–Trumbore intersection algorithm to determine if a ray
 * intersects with a triangle. It calculates intersection points based on the triangle's vertices.
 *
 * @param[in] rayOrigin The origin of the ray in 3D space.
 * @param[in] rayDir The direction of the ray in 3D space.
 * @param[in] tri The triangle defined by its three vertices in 3D space.
 * @param[out] t The distance from rayOrigin to the intersection point along rayDir.
 * @return True if the ray intersects the triangle; otherwise, false.
 */
__device__ bool intersectRayTriangle(const Vec3Device_t &rayOrigin, const Vec3Device_t &rayDir, const TriangleDevice_t &tri, float &t)
{
    const float EPSILON = 1e-6f; // Threshold to handle floating-point precision issues

    // Calculate edges of the triangle
    Vec3Device_t edge1, edge2, h, s, q;
    float a, f, u, v;

    // Edge from vertex v0 to v1
    edge1.x = tri.v1.x - tri.v0.x;
    edge1.y = tri.v1.y - tri.v0.y;
    edge1.z = tri.v1.z - tri.v0.z;

    // Edge from vertex v0 to v2
    edge2.x = tri.v2.x - tri.v0.x;
    edge2.y = tri.v2.y - tri.v0.y;
    edge2.z = tri.v2.z - tri.v0.z;

    // Calculate the cross product of rayDir and edge2
    h.x = rayDir.y * edge2.z - rayDir.z * edge2.y;
    h.y = rayDir.z * edge2.x - rayDir.x * edge2.z;
    h.z = rayDir.x * edge2.y - rayDir.y * edge2.x;

    // Calculate determinant
    a = edge1.x * h.x + edge1.y * h.y + edge1.z * h.z;

    // If determinant is near zero, the ray is parallel to the triangle
    if (fabsf(a) < EPSILON)
        return false;

    // Calculate f as the inverse of the determinant
    f = 1.0f / a;

    // Calculate vector from vertex v0 to ray origin
    s.x = rayOrigin.x - tri.v0.x;
    s.y = rayOrigin.y - tri.v0.y;
    s.z = rayOrigin.z - tri.v0.z;

    // Calculate u parameter and test bounds
    u = f * (s.x * h.x + s.y * h.y + s.z * h.z);
    if (u < 0.0f || u > 1.0f)
        return false;

    // Calculate cross product of s and edge1
    q.x = s.y * edge1.z - s.z * edge1.y;
    q.y = s.z * edge1.x - s.x * edge1.z;
    q.z = s.x * edge1.y - s.y * edge1.x;

    // Calculate v parameter and test bounds
    v = f * (rayDir.x * q.x + rayDir.y * q.y + rayDir.z * q.z);
    if (v < 0.0f || u + v > 1.0f)
        return false;

    // Calculate t, the distance from the ray origin to the intersection point
    t = f * (edge2.x * q.x + edge2.y * q.y + edge2.z * q.z);

    // If t is positive, the intersection is in the ray's forward direction
    if (t > EPSILON)
        return true;

    // No intersection
    return false;
}

#endif // !USE_CUDA

#endif // !INTERSECTIONS_CUH
