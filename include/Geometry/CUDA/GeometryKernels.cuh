#ifndef GEOMETRYKERNELS_CUH
#define GEOMETRYKERNELS_CUH

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include "Geometry/CUDA/GeometryDeviceTypes.cuh"
#include "Utilities/CUDAWarningSuppress.hpp"
#include "Utilities/PreprocessorUtils.hpp"

/**
 * @brief Kernel to check if points are inside a mesh represented by triangles
 * 
 * @param points Array of points to check
 * @param triangles Array of triangles representing the mesh
 * @param numPoints Number of points
 * @param numTriangles Number of triangles
 * @param results Output array indicating if each point is inside (1) or outside (0)
 */
START_CUDA_GLOBAL void checkPointsInsideMeshKernel(
    DevicePoint* points,
    DeviceTriangle* triangles,
    int numPoints,
    int numTriangles,
    int* results
);

/**
 * @brief Kernel to calculate distances from points to a mesh
 * 
 * @param points Array of points
 * @param triangles Array of triangles representing the mesh
 * @param numPoints Number of points
 * @param numTriangles Number of triangles
 * @param distances Output array of distances
 */
START_CUDA_GLOBAL void calculateDistancesToMeshKernel(
    DevicePoint* points,
    DeviceTriangle* triangles,
    int numPoints,
    int numTriangles,
    double* distances
);

/**
 * @brief Kernel to perform ray-triangle intersection tests
 * 
 * @param rays Array of rays
 * @param triangles Array of triangles
 * @param numRays Number of rays
 * @param numTriangles Number of triangles
 * @param hitResults Output array indicating hit (1) or miss (0) for each ray
 * @param hitDistances Output array of hit distances (if hit)
 */
START_CUDA_GLOBAL void rayTriangleIntersectionKernel(
    DeviceRay* rays,
    DeviceTriangle* triangles,
    int numRays,
    int numTriangles,
    int* hitResults,
    double* hitDistances
);

#endif // !USE_CUDA

#endif // !GEOMETRYKERNELS_CUH 
