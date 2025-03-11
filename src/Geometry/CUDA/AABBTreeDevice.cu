#ifdef USE_CUDA

#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <vector>

#include "Geometry/CUDA/AABBTreeDevice.cuh"
#include "Geometry/CUDA/GeometryKernels.cuh"
#include "Utilities/CUDA/DeviceUtils.cuh"
#include "Utilities/CUDAWarningSuppress.hpp"
#include "Utilities/LogMacros.hpp"

// Helper function to compute the bounding box of a range of triangles
DeviceAABB computeBoundingBox(const std::vector<DeviceTriangle> &triangles, size_t start, size_t end)
{
    DeviceAABB bbox;

    // Expand the bounding box to include all triangles in the range
    for (size_t i = start; i < end; ++i)
    {
        const DeviceTriangle &tri = triangles[i];
        bbox.expand(tri.v0);
        bbox.expand(tri.v1);
        bbox.expand(tri.v2);
    }

    return bbox;
}

START_CUDA_HOST_DEVICE void DeviceAABB::expand(DevicePoint const &point)
{
    min.x = fmin(min.x, point.x);
    min.y = fmin(min.y, point.y);
    min.z = fmin(min.z, point.z);

    max.x = fmax(max.x, point.x);
    max.y = fmax(max.y, point.y);
    max.z = fmax(max.z, point.z);
}

START_CUDA_HOST_DEVICE void DeviceAABB::expand(DeviceAABB const &box)
{
    expand(box.min);
    expand(box.max);
}

START_CUDA_HOST_DEVICE bool DeviceAABB::intersects(DeviceAABB const &box) const
{
    return (min.x <= box.max.x && max.x >= box.min.x) &&
           (min.y <= box.max.y && max.y >= box.min.y) &&
           (min.z <= box.max.z && max.z >= box.min.z);
}

START_CUDA_HOST_DEVICE bool DeviceAABB::intersects(const DeviceRay &ray) const noexcept
{
    // Ray-AABB intersection using slab method
    double t_min = 0.0;
    double t_max = ray.length;

    // Check intersection with x planes
    if (std::abs(ray.direction.x) < 1e-8)
    {
        // Ray is parallel to x planes
        if (ray.origin.x < min.x || ray.origin.x > max.x)
            return false;
    }
    else
    {
        double inv_dir_x = 1.0 / ray.direction.x;
        double t1 = (min.x - ray.origin.x) * inv_dir_x;
        double t2 = (max.x - ray.origin.x) * inv_dir_x;

        if (t1 > t2)
            std::swap(t1, t2);

        t_min = t1 > t_min ? t1 : t_min;
        t_max = t2 < t_max ? t2 : t_max;

        if (t_min > t_max)
            return false;
    }

    // Check intersection with y planes
    if (std::abs(ray.direction.y) < 1e-8)
    {
        // Ray is parallel to y planes
        if (ray.origin.y < min.y || ray.origin.y > max.y)
            return false;
    }
    else
    {
        double inv_dir_y = 1.0 / ray.direction.y;
        double t1 = (min.y - ray.origin.y) * inv_dir_y;
        double t2 = (max.y - ray.origin.y) * inv_dir_y;

        if (t1 > t2)
            std::swap(t1, t2);

        t_min = t1 > t_min ? t1 : t_min;
        t_max = t2 < t_max ? t2 : t_max;

        if (t_min > t_max)
            return false;
    }

    // Check intersection with z planes
    if (std::abs(ray.direction.z) < 1e-8)
    {
        // Ray is parallel to z planes
        if (ray.origin.z < min.z || ray.origin.z > max.z)
            return false;
    }
    else
    {
        double inv_dir_z = 1.0 / ray.direction.z;
        double t1 = (min.z - ray.origin.z) * inv_dir_z;
        double t2 = (max.z - ray.origin.z) * inv_dir_z;

        if (t1 > t2)
            std::swap(t1, t2);

        t_min = t1 > t_min ? t1 : t_min;
        t_max = t2 < t_max ? t2 : t_max;

        if (t_min > t_max)
            return false;
    }

    return true;
}

AABBTreeDevice::AABBTreeDevice()
    : d_nodes(nullptr), d_triangles(nullptr), num_nodes(0), num_triangles(0)
{
}

AABBTreeDevice::~AABBTreeDevice()
{
    if (d_nodes)
    {
        cudaFree(d_nodes);
        d_nodes = nullptr;
    }

    if (d_triangles)
    {
        cudaFree(d_triangles);
        d_triangles = nullptr;
    }
}

void AABBTreeDevice::initialize(const std::vector<DeviceTriangle> &triangles)
{
    if (triangles.empty())
    {
        ERRMSG("Cannot initialize AABB tree with empty triangle list");
        return;
    }

    // Clean up any existing data
    if (d_nodes)
    {
        cudaFree(d_nodes);
        d_nodes = nullptr;
    }

    if (d_triangles)
    {
        cudaFree(d_triangles);
        d_triangles = nullptr;
    }

    // Build the tree on the host
    buildTree(triangles);
}

void AABBTreeDevice::buildTree(const std::vector<DeviceTriangle> &triangles)
{
    num_triangles = static_cast<int>(triangles.size());

    // For simplicity, we'll create a flat tree where each node is a leaf containing one triangle
    // This is not optimal but will work for our purposes
    num_nodes = num_triangles;

    // Allocate host memory for nodes and triangles
    std::vector<DeviceAABBNode> h_nodes(num_nodes);

    // Create leaf nodes for each triangle
    for (int i = 0; i < num_triangles; ++i)
    {
        const DeviceTriangle &tri = triangles[i];

        // Compute AABB for the triangle
        DeviceAABB bbox;
        bbox.expand(tri.v0);
        bbox.expand(tri.v1);
        bbox.expand(tri.v2);

        // Create leaf node
        h_nodes[i].bounds = bbox;
        h_nodes[i].triangle_idx = i;
        h_nodes[i].left_child = -1;
        h_nodes[i].right_child = -1;
    }

    // Allocate device memory for nodes and triangles
    cudaError_t err = cudaMalloc(&d_nodes, num_nodes * sizeof(DeviceAABBNode));
    cuda_utils::check_cuda_err(err, "Failed to allocate device memory for AABB nodes");

    err = cudaMalloc(&d_triangles, num_triangles * sizeof(DeviceTriangle));
    cuda_utils::check_cuda_err(err, "Failed to allocate device memory for triangles");

    // Copy data to device
    err = cudaMemcpy(d_nodes, h_nodes.data(), num_nodes * sizeof(DeviceAABBNode), cudaMemcpyHostToDevice);
    cuda_utils::check_cuda_err(err, "Failed to copy AABB nodes to device");

    err = cudaMemcpy(d_triangles, triangles.data(), num_triangles * sizeof(DeviceTriangle), cudaMemcpyHostToDevice);
    cuda_utils::check_cuda_err(err, "Failed to copy triangles to device");

    LOGMSG(util::stringify("AABB tree built with ", num_nodes, " nodes and ", num_triangles, " triangles"));
}

bool AABBTreeDevice::any_intersection(const DeviceRay &ray, int &triangle_idx, double &distance) const
{
    if (!d_nodes || !d_triangles || num_nodes == 0)
    {
        ERRMSG("AABB tree not initialized");
        return false;
    }

    // Allocate device memory for results
    bool *d_has_intersection;
    int *d_triangle_idx;
    double *d_distance;

    cudaError_t err = cudaMalloc(&d_has_intersection, sizeof(bool));
    cuda_utils::check_cuda_err(err, "Failed to allocate device memory for intersection result");

    err = cudaMalloc(&d_triangle_idx, sizeof(int));
    cuda_utils::check_cuda_err(err, "Failed to allocate device memory for triangle index");

    err = cudaMalloc(&d_distance, sizeof(double));
    cuda_utils::check_cuda_err(err, "Failed to allocate device memory for distance");

    // Initialize results
    bool h_has_intersection = false;
    err = cudaMemcpy(d_has_intersection, &h_has_intersection, sizeof(bool), cudaMemcpyHostToDevice);
    cuda_utils::check_cuda_err(err, "Failed to initialize intersection result");

    triangle_idx = -1;
    err = cudaMemcpy(d_triangle_idx, &triangle_idx, sizeof(int), cudaMemcpyHostToDevice);
    cuda_utils::check_cuda_err(err, "Failed to initialize triangle index");

    distance = std::numeric_limits<double>::max();
    err = cudaMemcpy(d_distance, &distance, sizeof(double), cudaMemcpyHostToDevice);
    cuda_utils::check_cuda_err(err, "Failed to initialize distance");

    // Launch kernel to find intersection
    findRayTriangleIntersection<<<1, 1>>>(ray, d_nodes, d_triangles, 0, d_has_intersection, d_triangle_idx, d_distance);

    // Check for kernel errors
    err = cudaGetLastError();
    cuda_utils::check_cuda_err(err, "Failed to launch ray-triangle intersection kernel");

    // Wait for kernel to finish
    err = cudaDeviceSynchronize();
    cuda_utils::check_cuda_err(err, "Failed to synchronize after ray-triangle intersection kernel");

    // Copy results back to host
    err = cudaMemcpy(&h_has_intersection, d_has_intersection, sizeof(bool), cudaMemcpyDeviceToHost);
    cuda_utils::check_cuda_err(err, "Failed to copy intersection result from device");

    if (h_has_intersection)
    {
        err = cudaMemcpy(&triangle_idx, d_triangle_idx, sizeof(int), cudaMemcpyDeviceToHost);
        cuda_utils::check_cuda_err(err, "Failed to copy triangle index from device");

        err = cudaMemcpy(&distance, d_distance, sizeof(double), cudaMemcpyDeviceToHost);
        cuda_utils::check_cuda_err(err, "Failed to copy distance from device");
    }

    // Free device memory
    cudaFree(d_has_intersection);
    cudaFree(d_triangle_idx);
    cudaFree(d_distance);

    return h_has_intersection;
}

START_CUDA_GLOBAL void findRayTriangleIntersection(
    DeviceRay ray,
    DeviceAABBNode *nodes,
    DeviceTriangle *triangles,
    int root_idx,
    bool *has_intersection,
    int *triangle_idx,
    double *distance)
{
    // Simple linear search through all nodes (for now)
    // In a real implementation, we would traverse the tree recursively

    bool found = false;
    int found_idx = -1;
    double min_distance = 1e30;

    for (int i = 0; i < 1000; ++i)
    { // Limit to 1000 nodes to avoid infinite loops
        if (i >= root_idx)
        {
            DeviceAABBNode &node = nodes[i];

            // Check if ray intersects node's bounding box
            if (node.bounds.intersects(ray))
            {
                // If this is a leaf node, check for triangle intersection
                if (node.triangle_idx >= 0)
                {
                    DeviceTriangle &tri = triangles[node.triangle_idx];
                    double hit_distance;

                    if (cuda_kernels::rayTriangleIntersection(ray, tri, hit_distance))
                    {
                        if (hit_distance < min_distance)
                        {
                            min_distance = hit_distance;
                            found_idx = node.triangle_idx;
                            found = true;
                        }
                    }
                }
            }
        }
    }

    // Write results
    *has_intersection = found;
    if (found)
    {
        *triangle_idx = found_idx;
        *distance = min_distance;
    }
}

#endif // USE_CUDA
