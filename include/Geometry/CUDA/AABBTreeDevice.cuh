#ifndef AABBTREEDEVICE_CUH
#define AABBTREEDEVICE_CUH

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <vector>

#include "Geometry/CUDA/GeometryTypes.cuh"
#include "Geometry/GeometryTypes.hpp"
#include "Utilities/PreprocessorUtils.hpp"

/// @brief Axis-Aligned Bounding Box (AABB) structure.
struct AABBDevice_t
{
    Vec3Device_t min; ///< Minimum coordinates of the bounding box.
    Vec3Device_t max; ///< Maximum coordinates of the bounding box.

    /// @brief Expand the AABB to include a point.
    /// @param point The point to include.
    START_CUDA_HOST_DEVICE void expand(Vec3Device_t const &point);

    /// @brief Expand the AABB to include another AABB.
    /// @param box The AABB to include.
    START_CUDA_HOST_DEVICE void expand(AABBDevice_t const &box);

    /// @brief Check if this AABB intersects with another AABB.
    /// @param box The other AABB.
    /// @return True if the boxes intersect, false otherwise.
    START_CUDA_HOST_DEVICE bool intersects(AABBDevice_t const &box) const;
};

/// @brief Structure representing a node in the AABB tree.
struct AABBNodeDevice_t
{
    AABBDevice_t bbox; ///< Bounding box of the node.
    int left;          ///< Index of the left child (-1 if leaf).
    int right;         ///< Index of the right child (-1 if leaf).
    int triangleIdx;   ///< Index of the triangle (if leaf node).
};

/// @brief Class representing an AABB tree for collision detection.
/// The tree is stored as an array of nodes suitable for GPU traversal.
class AABBTreeDevice
{
public:
    /// @brief Construct the AABB tree from a list of triangles.
    /// @param triangles Vector of triangles.
    void build(std::vector<TriangleDevice_t> triangles);

    /**
     * @brief Recursively builds the Axis-Aligned Bounding Box (AABB) tree for a range of triangles.
     *
     * This function creates a binary tree structure for efficient spatial querying of triangle geometry.
     * The method computes bounding boxes for nodes and splits triangles recursively to construct the tree.
     *
     * @param triangles The vector of triangles for which the AABB tree is being constructed.
     * @param start The starting index (inclusive) of the range of triangles being processed.
     * @param end The ending index (exclusive) of the range of triangles being processed.
     * @return The index of the current node in the `hostNodes` vector, or -1 if the range is invalid.
     *
     * ### Method Details:
     * - If the range contains only one triangle (`end - start == 1`), a leaf node is created containing
     *   the triangle's index. The leaf node has no children.
     * - For ranges with more than one triangle, an interior node is created. The method determines
     *   the axis with the largest extent of the bounding box (e.g., x, y, or z axis) and sorts the
     *   triangles along that axis. The range is then split at the midpoint, and the left and right
     *   child nodes are built recursively.
     * - Each node stores a bounding box (`bbox`) that encloses all triangles in its range, and references
     *   to its left and right child nodes in the tree hierarchy.
     *
     * ### Bounding Box Calculation:
     * - The bounding box for the node is computed by encompassing all triangles in the range. This ensures
     *   that every triangle within the range is spatially represented by the node's bounding box.
     *
     * ### Sorting and Splitting:
     * - To achieve spatial locality and minimize tree depth, triangles are sorted based on their centroids
     *   along the axis with the largest extent of the bounding box.
     * - The range of triangles is split at the midpoint, balancing the distribution of triangles between
     *   the left and right child nodes.
     *
     * ### Node Indexing:
     * - The method returns the index of the node in the `hostNodes` vector. This index is used by
     *   parent nodes to reference their children.
     * - Leaf nodes store the index of the triangle they represent, while interior nodes store references
     *   to their left and right children.
     *
     * ### Example:
     * Suppose the input range includes triangles `triangles[start]` to `triangles[end - 1]`. The method:
     * 1. Computes a bounding box for the range.
     * 2. If only one triangle exists in the range, creates a leaf node.
     * 3. Otherwise, splits the range into two halves, builds left and right subtrees, and returns the
     *    index of the interior node created.
     *
     * @note This method assumes the `hostNodes` vector has sufficient capacity to store nodes.
     * @note Sorting modifies the order of triangles in the input vector within the specified range.
     *
     * @exception None. If an invalid range is provided (`start >= end`), the method gracefully returns -1.
     */
    int buildRecursive(std::vector<TriangleDevice_t> triangles, size_t start, size_t end);

    /// @brief Get the number of nodes in the tree.
    /// @return Number of nodes.
    size_t getNodeCount() const;

    /// @brief Get the device pointer to the nodes array.
    /// @return Device pointer to nodes.
    AABBNodeDevice_t *getDeviceNodes() const;

    /// @brief Free the allocated device memory.
    void freeDeviceMemory();

private:
    std::vector<AABBNodeDevice_t> hostNodes; ///< Nodes stored on host.
    AABBNodeDevice_t *deviceNodes;           ///< Nodes stored on device.
};

#endif // !USE_CUDA

#endif // AABBTREEDEVICE_CUH
