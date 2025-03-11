#ifndef AABBTREEDEVICE_CUH
#define AABBTREEDEVICE_CUH

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <vector>

#include "Geometry/CUDA/GeometryDeviceTypes.cuh"
#include "Utilities/CUDAWarningSuppress.hpp"
#include "Utilities/PreprocessorUtils.hpp"

/// @brief Axis-Aligned Bounding Box (AABB) structure.
struct DeviceAABB
{
    DevicePoint min;
    DevicePoint max;

    START_CUDA_HOST_DEVICE DeviceAABB() noexcept
    {
        min.x = min.y = min.z = std::numeric_limits<double>::infinity();
        max.x = max.y = max.z = -std::numeric_limits<double>::infinity();
    }

    START_CUDA_HOST_DEVICE DeviceAABB(double min_x, double min_y, double min_z,
                                      double max_x, double max_y, double max_z) noexcept
    {
        min.x = min_x;
        min.y = min_y;
        min.z = min_z;
        max.x = max_x;
        max.y = max_y;
        max.z = max_z;
    }

    START_CUDA_HOST_DEVICE void expand(DevicePoint const &point);
    START_CUDA_HOST_DEVICE void expand(DeviceAABB const &box);
    START_CUDA_HOST_DEVICE bool intersects(DeviceAABB const &box) const;
    START_CUDA_HOST_DEVICE bool intersects(const DeviceRay &ray) const noexcept;
};

/// @brief Structure representing a node in the AABB tree.
struct DeviceAABBNode
{
    DeviceAABB bounds; ///< Bounding box of the node.
    int triangle_idx;  // Index of the triangle, or -1 if this is an internal node
    int left_child;    // Index of the left child, or -1 if this is a leaf
    int right_child;   // Index of the right child, or -1 if this is a leaf

    START_CUDA_HOST_DEVICE DeviceAABBNode() noexcept
        : triangle_idx(-1), left_child(-1), right_child(-1) {}
};

/// @brief Class representing an AABB tree for collision detection.
/// The tree is stored as an array of nodes suitable for GPU traversal.
class AABBTreeDevice
{
private:
    DeviceAABBNode *d_nodes;
    DeviceTriangle *d_triangles;
    int num_nodes;
    int num_triangles;

    void buildTree(const std::vector<DeviceTriangle> &triangles);

public:
    AABBTreeDevice();
    ~AABBTreeDevice();

    void initialize(const std::vector<DeviceTriangle> &triangles);

    bool any_intersection(const DeviceRay &ray, int &triangle_idx, double &distance) const;

    DeviceAABBNode *getDeviceNodes() const { return d_nodes; }
    DeviceTriangle *getDeviceTriangles() const { return d_triangles; }
    int getNumNodes() const { return num_nodes; }
    int getNumTriangles() const { return num_triangles; }
};

START_CUDA_GLOBAL void findRayTriangleIntersection(
    DeviceRay ray,
    DeviceAABBNode *nodes,
    DeviceTriangle *triangles,
    int root_idx,
    bool *has_intersection,
    int *triangle_idx,
    double *distance);

#endif // !USE_CUDA

#endif // !AABBTREEDEVICE_CUH
