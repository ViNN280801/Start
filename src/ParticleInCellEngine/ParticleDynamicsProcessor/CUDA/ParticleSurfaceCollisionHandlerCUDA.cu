#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>

#include "Geometry/CUDA/AABBTreeDevice.cuh"
#include "Geometry/CUDA/GeometryKernels.cuh"
#include "ParticleInCellEngine/ParticleDynamicsProcessor/CUDA/ParticleSurfaceCollisionHandlerCUDA.cuh"
#include "Utilities/CUDA/DeviceUtils.cuh"
#include "Utilities/CUDAWarningSuppress.hpp"
#include "Utilities/LogMacros.hpp"

AABBTreeDevice ParticleSurfaceCollisionHandlerCUDA::s_aabbTree;

void ParticleSurfaceCollisionHandlerCUDA::initialize(const std::vector<DeviceTriangle> &triangles)
{
    LOGMSG("Initializing CUDA surface collision handler");
    s_aabbTree.initialize(triangles);
}

void ParticleSurfaceCollisionHandlerCUDA::processCollisions(
    ParticleDevice_t *particles,
    size_t num_particles,
    DevicePoint *previous_positions,
    int *settled_particles,
    DevicePoint *collision_points,
    int *collision_triangles,
    int *num_collisions)
{
    if (num_particles == 0)
    {
        LOGMSG("No particles to process for collisions");
        return;
    }

    // Initialize collision counter
    int zero = 0;
    cudaError_t err = cudaMemcpy(num_collisions, &zero, sizeof(int), cudaMemcpyHostToDevice);
    START_CHECK_CUDA_ERROR(err, "Failed to initialize collision counter");

    // Calculate grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_particles + threadsPerBlock - 1) / threadsPerBlock;

    LOGMSG(util::stringify("Launching collision detection kernel with ", blocksPerGrid, " blocks and ", threadsPerBlock, " threads per block"));

    // Launch kernel
    detectCollisionsKernel<<<blocksPerGrid, threadsPerBlock>>>(
        particles,
        num_particles,
        previous_positions,
        s_aabbTree.getDeviceNodes(),
        s_aabbTree.getDeviceTriangles(),
        s_aabbTree.getNumTriangles(),
        settled_particles,
        collision_points,
        collision_triangles,
        num_collisions);

    // Check for kernel errors
    err = cudaGetLastError();
    START_CHECK_CUDA_ERROR(err, "Failed to launch collision detection kernel");

    // Wait for kernel to finish
    err = cudaDeviceSynchronize();
    START_CHECK_CUDA_ERROR(err, "Failed to synchronize after collision detection kernel");

    // Get number of collisions (for logging)
    int host_num_collisions = 0;
    err = cudaMemcpy(&host_num_collisions, num_collisions, sizeof(int), cudaMemcpyDeviceToHost);
    START_CHECK_CUDA_ERROR(err, "Failed to copy collision count from device");

    LOGMSG(util::stringify("Detected ", host_num_collisions, " collisions"));
}

void ParticleSurfaceCollisionHandlerCUDA::cleanup()
{
    // The AABBTreeDevice destructor will handle cleanup
}

extern "C" __global__ void detectCollisionsKernel(
    ParticleDevice_t *particles,
    size_t num_particles,
    DevicePoint *previous_positions,
    DeviceAABBNode *aabb_nodes,
    DeviceTriangle *triangles,
    int num_triangles,
    int *settled_particles,
    DevicePoint *collision_points,
    int *collision_triangles,
    int *num_collisions)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_particles)
    {
        return;
    }

    // Skip already settled particles
    if (settled_particles[idx] != 0)
    {
        return;
    }

    ParticleDevice_t &particle = particles[idx];
    DevicePoint &prev_pos = previous_positions[idx];

    // Create ray from previous to current position
    DeviceRay ray;
    ray.origin = prev_pos;
    ray.direction = DevicePoint(
        particle.x - prev_pos.x,
        particle.y - prev_pos.y,
        particle.z - prev_pos.z);

    // Calculate ray length
    ray.length = sqrt(
        ray.direction.x * ray.direction.x +
        ray.direction.y * ray.direction.y +
        ray.direction.z * ray.direction.z);

    // Skip if ray is too short (particle didn't move)
    if (ray.length < 1e-10)
    {
        return;
    }

    // Normalize direction
    ray.direction.x /= ray.length;
    ray.direction.y /= ray.length;
    ray.direction.z /= ray.length;

    // Check for collisions with all triangles
    bool found_collision = false;
    int collision_triangle_idx = -1;
    double min_distance = ray.length; // Only consider intersections within the ray length
    DevicePoint intersection_pt;

    for (int i = 0; i < num_triangles; ++i)
    {
        DeviceTriangle &tri = triangles[i];
        double distance;

        if (cuda_kernels::rayTriangleIntersection(ray, tri, distance))
        {
            // Only consider intersections within the ray length and closer than any previous intersection
            if (distance < min_distance && distance > 0)
            {
                min_distance = distance;
                collision_triangle_idx = i;
                found_collision = true;

                // Calculate intersection point
                intersection_pt.x = ray.origin.x + ray.direction.x * distance;
                intersection_pt.y = ray.origin.y + ray.direction.y * distance;
                intersection_pt.z = ray.origin.z + ray.direction.z * distance;
            }
        }
    }

    if (found_collision)
    {
        // Mark particle as settled
        settled_particles[idx] = 1;

        // Record collision
        int collision_idx = atomicAdd(num_collisions, 1);

        // Store collision data
        collision_points[collision_idx] = intersection_pt;
        collision_triangles[collision_idx] = collision_triangle_idx;

        // Update particle position to the intersection point
        particle.x = intersection_pt.x;
        particle.y = intersection_pt.y;
        particle.z = intersection_pt.z;

        // Set velocity to zero (particle has settled)
        particle.vx = 0.0;
        particle.vy = 0.0;
        particle.vz = 0.0;
    }
}

#endif // USE_CUDA