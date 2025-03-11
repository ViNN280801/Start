#ifndef PARTICLESURFACECOLLISIONHANDLERCUDA_CUH
#define PARTICLESURFACECOLLISIONHANDLERCUDA_CUH

#ifdef USE_CUDA

#include <cuda_runtime.h>

#include "Geometry/CUDA/AABBTreeDevice.cuh"
#include "Geometry/CUDA/GeometryDeviceTypes.cuh"
#include "Particle/CUDA/ParticleDevice.cuh"
#include "Utilities/CUDAWarningSuppress.hpp"
#include "Utilities/PreprocessorUtils.hpp"

struct DeviceCollisionResult
{
    bool has_collision;
    int triangle_idx;
    double distance;
    DevicePoint intersection_point;
    
    START_CUDA_HOST_DEVICE DeviceCollisionResult() noexcept
        : has_collision(false), triangle_idx(-1), distance(0.0) {}
};

class ParticleSurfaceCollisionHandlerCUDA
{
public:
    static void initialize(const std::vector<DeviceTriangle>& triangles);
    
    static void processCollisions(
        ParticleDevice_t* particles,
        size_t num_particles,
        DevicePoint* previous_positions,
        int* settled_particles,
        DevicePoint* collision_points,
        int* collision_triangles,
        int* num_collisions
    );
    
    static void cleanup();
    
private:
    static AABBTreeDevice s_aabbTree;
};

START_CUDA_GLOBAL void detectCollisionsKernel(
    ParticleDevice_t* particles,
    size_t num_particles,
    DevicePoint* previous_positions,
    DeviceAABBNode* aabb_nodes,
    DeviceTriangle* triangles,
    int num_triangles,
    int* settled_particles,
    DevicePoint* collision_points,
    int* collision_triangles,
    int* num_collisions
);

#endif // USE_CUDA

#endif // PARTICLESURFACECOLLISIONHANDLERCUDA_CUH 