#ifdef USE_CUDA

#include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "Geometry/CUDA/AABBTreeDevice.cuh"

// Helper function to compute the bounding box of a range of triangles
AABBDevice_t computeBoundingBox(std::vector<TriangleDevice_t> triangles, size_t start, size_t end)
{
    AABBDevice_t bbox;

    // Initialize bbox to extreme values
    bbox.min = {std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity()};

    bbox.max = {-std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity()};

    // Expand the bounding box to include all triangles in the range
    for (size_t i = start; i < end; ++i)
    {
        const TriangleDevice_t &tri = triangles[i];
        bbox.expand(tri.v0);
        bbox.expand(tri.v1);
        bbox.expand(tri.v2);
    }

    return bbox;
}

START_CUDA_HOST_DEVICE void AABBDevice_t::expand(Vec3Device_t const &point)
{
    min.x = fminf(min.x, point.x);
    min.y = fminf(min.y, point.y);
    min.z = fminf(min.z, point.z);

    max.x = fmaxf(max.x, point.x);
    max.y = fmaxf(max.y, point.y);
    max.z = fmaxf(max.z, point.z);
}

START_CUDA_HOST_DEVICE void AABBDevice_t::expand(AABBDevice_t const &box)
{
    expand(box.min);
    expand(box.max);
}

START_CUDA_HOST_DEVICE bool AABBDevice_t::intersects(AABBDevice_t const &box) const
{
    return (min.x <= box.max.x && max.x >= box.min.x) &&
           (min.y <= box.max.y && max.y >= box.min.y) &&
           (min.z <= box.max.z && max.z >= box.min.z);
}

void AABBTreeDevice::build(std::vector<TriangleDevice_t> triangles)
{
    hostNodes.clear();
    hostNodes.reserve(2 * triangles.size()); // Reserve memory to prevent reallocations

    // Start building the tree recursively
    buildRecursive(triangles, 0, triangles.size());

    // Copy nodes to device memory
    size_t nodeSize = hostNodes.size() * sizeof(AABBNodeDevice_t);
    cudaMalloc(&deviceNodes, nodeSize);
    cudaMemcpy(deviceNodes, hostNodes.data(), nodeSize, cudaMemcpyHostToDevice);
}

int AABBTreeDevice::buildRecursive(std::vector<TriangleDevice_t> triangles, size_t start, size_t end)
{
    if (start >= end)
        return -1; // Invalid range, no node to create

    AABBNodeDevice_t node;
    node.left = -1;
    node.right = -1;
    node.triangleIdx = -1;

    // Compute the bounding box for the current node
    node.bbox = computeBoundingBox(triangles, start, end);

    // Current node index in the hostNodes vector
    int currentIndex = hostNodes.size();
    hostNodes.push_back(node);

    if (end - start == 1)
    {
        // Leaf node: store the triangle index
        hostNodes[currentIndex].triangleIdx = static_cast<int>(start);
        return currentIndex;
    }
    else
    {
        // Interior node: need to split the range
        // Determine the axis with the largest extent
        Vec3Device_t extent;
        extent.x = node.bbox.max.x - node.bbox.min.x;
        extent.y = node.bbox.max.y - node.bbox.min.y;
        extent.z = node.bbox.max.z - node.bbox.min.z;

        int axis = 0; // 0: x-axis, 1: y-axis, 2: z-axis
        if (extent.y > extent.x)
            axis = 1;
        if (extent.z > extent.y && extent.z > extent.x)
            axis = 2;

        // Sort triangles based on the centroid along the chosen axis
        std::vector<size_t> indices(end - start);
        for (size_t i = 0; i < indices.size(); ++i)
            indices[i] = start + i;

        std::sort(indices.begin(), indices.end(),
            [&triangles, axis](size_t a, size_t b)
            {
                float centroidA, centroidB;
                const TriangleDevice_t &triA = triangles[a];
                const TriangleDevice_t &triB = triangles[b];

                if (axis == 0) // x-axis
                {
                    centroidA = (triA.v0.x + triA.v1.x + triA.v2.x) / 3.0f;
                    centroidB = (triB.v0.x + triB.v1.x + triB.v2.x) / 3.0f;
                }
                else if (axis == 1) // y-axis
                {
                    centroidA = (triA.v0.y + triA.v1.y + triA.v2.y) / 3.0f;
                    centroidB = (triB.v0.y + triB.v1.y + triB.v2.y) / 3.0f;
                }
                else // z-axis
                {
                    centroidA = (triA.v0.z + triA.v1.z + triA.v2.z) / 3.0f;
                    centroidB = (triB.v0.z + triB.v1.z + triB.v2.z) / 3.0f;
                }

                return centroidA < centroidB;
            });

        // Rearrange triangles according to the sorted indices
        std::vector<TriangleDevice_t> sortedTriangles(end - start);
        for (size_t i = 0; i < indices.size(); ++i)
            sortedTriangles[i] = triangles[indices[i]];

        // Replace the original triangles in the range with the sorted ones
        std::copy(sortedTriangles.begin(), sortedTriangles.end(), triangles.begin() + start);

        // Compute the midpoint for splitting
        size_t mid = start + (end - start) / 2;

        // Recursively build left and right subtrees
        int leftChild = buildRecursive(triangles, start, mid);
        int rightChild = buildRecursive(triangles, mid, end);

        // Update the current node with child indices
        hostNodes[currentIndex].left = leftChild;
        hostNodes[currentIndex].right = rightChild;

        return currentIndex;
    }
}

size_t AABBTreeDevice::getNodeCount() const { return hostNodes.size(); }

AABBNodeDevice_t *AABBTreeDevice::getDeviceNodes() const { return deviceNodes; }

void AABBTreeDevice::freeDeviceMemory()
{
    if (deviceNodes)
    {
        cudaFree(deviceNodes);
        deviceNodes = nullptr;
    }
}

#endif // !USE_CUDA
