#ifdef USE_CUDA

#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>

#include "Geometry/CUDA/GeometryKernels.cuh"
#include "Utilities/CUDA/DeviceUtils.cuh"
#include "Utilities/LogMacros.hpp"

// Implementation of the kernel to check if points are inside a mesh
extern "C" __global__ void checkPointsInsideMeshKernel(
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
        if (cuda_kernels::rayTriangleIntersection(ray, triangles[i], distance))
        {
            intersectionCount++;
        }
    }

    // If odd number of intersections, point is inside
    results[pointIdx] = (intersectionCount % 2 == 1) ? 1 : 0;
}

// Implementation of the kernel to calculate distances from points to a mesh
extern "C" __global__ void calculateDistancesToMeshKernel(
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
extern "C" __global__ void rayTriangleIntersectionKernel(
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
        if (cuda_kernels::rayTriangleIntersection(ray, triangles[i], distance))
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