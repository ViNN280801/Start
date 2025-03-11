#ifndef RAYTRIANGLE_INTERSECTION_CUH
#define RAYTRIANGLE_INTERSECTION_CUH

#ifdef USE_CUDA

#include "Geometry/CUDA/GeometryDeviceTypes.cuh"
#include "Utilities/CUDAWarningSuppress.hpp"
#include "Utilities/PreprocessorUtils.hpp"

namespace cuda_kernels
{
    /**
     * @brief Checks if a ray intersects with a triangle (Möller–Trumbore algorithm)
     *
     * @param ray The ray to check
     * @param triangle The triangle to check
     * @param distance Output parameter for the distance to the intersection point
     * @return true If the ray intersects the triangle
     * @return false If the ray does not intersect the triangle
     */
    START_CUDA_HOST_DEVICE __device__ __forceinline__ bool rayTriangleIntersection(
        const DeviceRay &ray,
        const DeviceTriangle &triangle,
        double &distance)
    {
        // Möller–Trumbore algorithm for ray-triangle intersection
        const double EPSILON = 1e-8;

        // Edge vectors
        DevicePoint edge1, edge2;
        edge1.x = triangle.v1.x - triangle.v0.x;
        edge1.y = triangle.v1.y - triangle.v0.y;
        edge1.z = triangle.v1.z - triangle.v0.z;

        edge2.x = triangle.v2.x - triangle.v0.x;
        edge2.y = triangle.v2.y - triangle.v0.y;
        edge2.z = triangle.v2.z - triangle.v0.z;

        // Calculate determinant
        DevicePoint h;
        h.x = ray.direction.y * edge2.z - ray.direction.z * edge2.y;
        h.y = ray.direction.z * edge2.x - ray.direction.x * edge2.z;
        h.z = ray.direction.x * edge2.y - ray.direction.y * edge2.x;

        double det = edge1.x * h.x + edge1.y * h.y + edge1.z * h.z;

        // If determinant is near zero, ray lies in plane of triangle or ray is parallel to plane of triangle
        if (det > -EPSILON && det < EPSILON)
            return false;

        double inv_det = 1.0 / det;

        // Calculate u parameter
        DevicePoint s;
        s.x = ray.origin.x - triangle.v0.x;
        s.y = ray.origin.y - triangle.v0.y;
        s.z = ray.origin.z - triangle.v0.z;

        double u = inv_det * (s.x * h.x + s.y * h.y + s.z * h.z);

        // Check if intersection is outside the triangle
        if (u < 0.0 || u > 1.0)
            return false;

        // Calculate v parameter
        DevicePoint q;
        q.x = s.y * edge1.z - s.z * edge1.y;
        q.y = s.z * edge1.x - s.x * edge1.z;
        q.z = s.x * edge1.y - s.y * edge1.x;

        double v = inv_det * (ray.direction.x * q.x + ray.direction.y * q.y + ray.direction.z * q.z);

        // Check if intersection is outside the triangle
        if (v < 0.0 || u + v > 1.0)
            return false;

        // Calculate t, ray intersection distance
        double t = inv_det * (edge2.x * q.x + edge2.y * q.y + edge2.z * q.z);

        // Check if intersection is behind the ray origin
        if (t < EPSILON)
            return false;

        // Set the distance to the intersection point
        distance = t;

        return true;
    }
}

#endif // !USE_CUDA
#endif // !RAYTRIANGLE_INTERSECTION_CUH
