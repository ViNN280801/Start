#ifdef USE_CUDA

#include <cmath>

#include "Geometry/CUDA/GeometryKernels.cuh"
#include "Utilities/CUDAWarningSuppress.hpp"
#include "Utilities/PreprocessorUtils.hpp"

// Helper function to check if a ray intersects a triangle
START_CUDA_DEVICE bool rayTriangleIntersection(
    const DeviceRay &ray,
    const DeviceTriangle &triangle,
    double &distance)
{
    // Moller-Trumbore algorithm
    const double EPSILON = 1e-8;

    // Edge vectors
    double e1x = triangle.v1.x - triangle.v0.x;
    double e1y = triangle.v1.y - triangle.v0.y;
    double e1z = triangle.v1.z - triangle.v0.z;

    double e2x = triangle.v2.x - triangle.v0.x;
    double e2y = triangle.v2.y - triangle.v0.y;
    double e2z = triangle.v2.z - triangle.v0.z;

    // Begin calculating determinant - also used to calculate U parameter
    double px = ray.direction.y * e2z - ray.direction.z * e2y;
    double py = ray.direction.z * e2x - ray.direction.x * e2z;
    double pz = ray.direction.x * e2y - ray.direction.y * e2x;

    // Determinant
    double det = e1x * px + e1y * py + e1z * pz;

    // Check if ray is parallel to triangle
    if (fabs(det) < EPSILON)
    {
        return false;
    }

    double inv_det = 1.0 / det;

    // Calculate vector from triangle origin to ray origin
    double tx = ray.origin.x - triangle.v0.x;
    double ty = ray.origin.y - triangle.v0.y;
    double tz = ray.origin.z - triangle.v0.z;

    // Calculate U parameter
    double u = (tx * px + ty * py + tz * pz) * inv_det;

    // Check if intersection is outside triangle
    if (u < 0.0 || u > 1.0)
    {
        return false;
    }

    // Calculate V parameter
    double qx = ty * e1z - tz * e1y;
    double qy = tz * e1x - tx * e1z;
    double qz = tx * e1y - ty * e1x;

    double v = (ray.direction.x * qx + ray.direction.y * qy + ray.direction.z * qz) * inv_det;

    // Check if intersection is outside triangle
    if (v < 0.0 || u + v > 1.0)
    {
        return false;
    }

    // Calculate distance
    distance = (e2x * qx + e2y * qy + e2z * qz) * inv_det;

    // Check if intersection is behind ray origin
    if (distance < 0.0)
    {
        return false;
    }

    return true;
}

// Implementation of the kernel to check if points are inside a mesh
START_CUDA_GLOBAL void checkPointsInsideMeshKernel(
    DevicePoint *points,
    DeviceTriangle *triangles,
    int numPoints,
    int numTriangles,
    int *results)
{
    int pointIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pointIdx >= numPoints)
    {
        return;
    }

    // Ray casting algorithm: count intersections with mesh
    // If odd number of intersections, point is inside

    DevicePoint &point = points[pointIdx];

    // Create a ray in a fixed direction (e.g., positive X)
    DeviceRay ray;
    ray.origin = point;
    ray.direction = DevicePoint(1.0, 0.0, 0.0);
    ray.length = 1e6; // Very long ray

    int intersectionCount = 0;
    double distance;

    for (int i = 0; i < numTriangles; i++)
    {
        if (rayTriangleIntersection(ray, triangles[i], distance))
        {
            intersectionCount++;
        }
    }

    // If odd number of intersections, point is inside
    results[pointIdx] = (intersectionCount % 2 == 1) ? 1 : 0;
}

// Implementation of the kernel to calculate distances from points to a mesh
START_CUDA_GLOBAL void calculateDistancesToMeshKernel(
    DevicePoint *points,
    DeviceTriangle *triangles,
    int numPoints,
    int numTriangles,
    double *distances)
{
    int pointIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pointIdx >= numPoints)
    {
        return;
    }

    DevicePoint &point = points[pointIdx];
    double minDistance = 1e30; // Very large initial distance

    for (int i = 0; i < numTriangles; i++)
    {
        DeviceTriangle &triangle = triangles[i];

        // Calculate distance from point to triangle
        // This is a simplified implementation - in practice, you'd use a more
        // sophisticated algorithm to compute point-triangle distance

        // For now, just compute distance to each vertex and take minimum
        double d1 = point.distance(triangle.v0);
        double d2 = point.distance(triangle.v1);
        double d3 = point.distance(triangle.v2);

        double triangleMinDist = fmin(d1, fmin(d2, d3));
        minDistance = fmin(minDistance, triangleMinDist);
    }

    distances[pointIdx] = minDistance;
}

// Implementation of the kernel to perform ray-triangle intersection tests
START_CUDA_GLOBAL void rayTriangleIntersectionKernel(
    DeviceRay *rays,
    DeviceTriangle *triangles,
    int numRays,
    int numTriangles,
    int *hitResults,
    double *hitDistances)
{
    int rayIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (rayIdx >= numRays)
    {
        return;
    }

    DeviceRay &ray = rays[rayIdx];
    bool hit = false;
    double minDistance = 1e30; // Very large initial distance

    for (int i = 0; i < numTriangles; i++)
    {
        double distance;
        if (rayTriangleIntersection(ray, triangles[i], distance))
        {
            if (distance < minDistance)
            {
                minDistance = distance;
                hit = true;
            }
        }
    }

    hitResults[rayIdx] = hit ? 1 : 0;
    hitDistances[rayIdx] = hit ? minDistance : -1.0;
}

#endif // USE_CUDA