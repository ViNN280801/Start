#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <vector>

#include "Particle/CUDA/ParticleDevice.cuh"
#include "Particle/CUDA/ParticleDeviceMemoryConverter.cuh"
#include "ParticleInCellEngine/ParticleDynamicsProcessor/CUDA/ParticleDynamicsProcessorCUDA.cuh"
#include "ParticleInCellEngine/ParticleDynamicsProcessor/CUDA/ParticleKernelHelpers.cuh"
#include "ParticleInCellEngine/ParticleDynamicsProcessor/CUDA/ParticleSurfaceCollisionHandlerCUDA.cuh"
#include "ParticleInCellEngine/ParticleDynamicsProcessor/ParticleMovementTracker.hpp"
#include "ParticleInCellEngine/ParticleDynamicsProcessor/ParticleSettler.hpp"
#include "ParticleInCellEngine/ParticleDynamicsProcessor/ParticleSurfaceCollisionHandler.hpp"
#include "Utilities/CUDA/DeviceUtils.cuh"
#include "Utilities/CUDAWarningSuppress.hpp"
#include "Utilities/LogMacros.hpp"

// Maximum number of threads per block
constexpr int MAX_THREADS_PER_BLOCK = 256;

/**
 * @brief Main CUDA kernel for particle dynamics processing
 *
 * This kernel processes particles on the GPU, updating their positions and velocities
 * based on gas collisions and other physics.
 *
 * @param particles Array of particles to process
 * @param count Number of particles
 * @param params Kernel parameters including time step, etc.
 */
START_CUDA_WARNING_SUPPRESS
extern "C" __global__ void particleDynamicsKernel(ParticleDevice_t *particles, size_t count, ParticleDynamicsKernelParams params)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= count)
        return;

    ParticleDevice_t &particle = particles[idx];

    // 1. Update position
    cuda_kernels::updateParticlePosition(particle, params.timeStep);

    // 2. Simulate collisions with gas molecules
    cuda_kernels::collideWithGas(particle,
                                 params.gasType,
                                 params.gasConcentration,
                                 params.timeStep,
                                 params.scatteringModelType);

    // Note: Surface collision and settling cannot be handled directly in the kernel
    // as they require complex data structures like the surface mesh and particle tracking
    // These will be handled on the CPU after the kernel completes
}
END_CUDA_WARNING_SUPPRESS

ParticleDeviceArray ParticleDynamicsProcessorCUDA::prepareParticlesForGPU(ParticleVector &particles)
{
    std::vector<ParticleDevice_t> deviceParticles;
    deviceParticles.reserve(particles.size());

    // Convert host particles to device format
    for (Particle const &particle : particles)
    {
        ParticleDevice_t deviceParticle;
        deviceParticle.id = particle.getId();
        deviceParticle.type = static_cast<int>(particle.getType());
        deviceParticle.x = particle.getX();
        deviceParticle.y = particle.getY();
        deviceParticle.z = particle.getZ();
        deviceParticle.vx = particle.getVx();
        deviceParticle.vy = particle.getVy();
        deviceParticle.vz = particle.getVz();
        deviceParticle.energy = particle.getEnergy_J();

        deviceParticles.push_back(deviceParticle);
    }

    // Allocate and copy to device memory
    return ParticleDeviceMemoryConverter::allocateDeviceArrayMemory(deviceParticles);
}

void ParticleDynamicsProcessorCUDA::updateHostParticles(ParticleDeviceArray &deviceParticles, ParticleVector &hostParticles)
{
    // Copy data back from device to host
    std::vector<Particle> updatedParticles = ParticleDeviceMemoryConverter::copyToHost(deviceParticles);

    // Update host particles with new positions and velocities
    if (updatedParticles.size() == hostParticles.size())
    {
        for (size_t i = 0; i < hostParticles.size(); ++i)
        {
            // Only copy position, velocity, and energy
            // Do not overwrite other particle properties
            hostParticles[i].setCentre(
                updatedParticles[i].getX(),
                updatedParticles[i].getY(),
                updatedParticles[i].getZ());
            hostParticles[i].setVelocity(
                updatedParticles[i].getVx(),
                updatedParticles[i].getVy(),
                updatedParticles[i].getVz());
            hostParticles[i].setEnergy_J(updatedParticles[i].getEnergy_J());
        }
    }
}

constexpr int ParticleDynamicsProcessorCUDA::convertScatteringModelToInt(std::string_view scatteringModel) noexcept
{
    if (scatteringModel == "HS")
        return 0;
    else if (scatteringModel == "VHS")
        return 1;
    else if (scatteringModel == "VSS")
        return 2;
    return 0; // Default to HS
}

constexpr int ParticleDynamicsProcessorCUDA::convertGasNameToParticleType(std::string_view gasName) noexcept
{
    if (gasName == "O2")
        return static_cast<int>(constants::particle_types::O2);
    else if (gasName == "Ar")
        return static_cast<int>(constants::particle_types::Ar);
    else if (gasName == "Ne")
        return static_cast<int>(constants::particle_types::Ne);
    else if (gasName == "He")
        return static_cast<int>(constants::particle_types::He);

    return static_cast<int>(constants::particle_types::Unknown);
}

void ParticleDynamicsProcessorCUDA::process(double timeMoment,
                                            double timeStep,
                                            ParticleVector &particles,
                                            std::shared_ptr<CubicGrid> cubicGrid,
                                            std::shared_ptr<GSMAssembler> gsmAssembler,
                                            SurfaceMesh &surfaceMesh,
                                            ParticleTrackerMap &particleTracker,
                                            ParticlesIDSet &settledParticlesIds,
                                            std::shared_mutex &sh_mutex_settledParticlesCounterMap,
                                            std::mutex &mutex_particlesMovementMap,
                                            ParticleMovementMap &particleMovementMap,
                                            StopSubject &stopSubject,
                                            std::string_view scatteringModel,
                                            std::string_view gasName,
                                            double gasConcentration)
{
    try
    {
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        START_CHECK_CUDA_ERROR(err, "Error querying CUDA device count");
        if (deviceCount == 0) {
            throw std::runtime_error("No CUDA devices found");
        }
        
        cudaDeviceProp deviceProp;
        err = cudaGetDeviceProperties(&deviceProp, 0);
        START_CHECK_CUDA_ERROR(err, "Error getting CUDA device properties");
        
        LOGMSG(util::stringify("Using CUDA device: ", deviceProp.name, 
               " with compute capability ", deviceProp.major, ".", deviceProp.minor));

        // Skip if there are no particles to process
        if (particles.empty())
            return;

        // 1. Prepare particles for GPU processing
        auto deviceParticles = prepareParticlesForGPU(particles);

        // 2. Record previous positions for movement tracking
        std::vector<DevicePoint> previousPositions;
        previousPositions.reserve(particles.size());
        for (auto &particle : particles)
        {
            Point center = particle.getCentre();
            previousPositions.push_back(DevicePoint(center.x(), center.y(), center.z()));
        }

        // 3. Prepare kernel parameters
        ParticleDynamicsKernelParams params;
        params.timeMoment = timeMoment;
        params.timeStep = timeStep;
        params.gasConcentration = gasConcentration;
        params.scatteringModelType = convertScatteringModelToInt(scatteringModel);
        params.gasType = convertGasNameToParticleType(gasName);

        // 4. Launch kernel to update particle positions and handle gas collisions
        int numThreads = MAX_THREADS_PER_BLOCK;
        int numBlocks = static_cast<int>((particles.size() + numThreads - 1) / numThreads);
        
        LOGMSG(util::stringify("Launching particle dynamics kernel with ", numBlocks, 
               " blocks and ", numThreads, " threads per block for ", particles.size(), " particles"));

        particleDynamicsKernel<<<numBlocks, numThreads>>>(
            deviceParticles.get(),
            particles.size(),
            params);

        // Check for kernel errors
        err = cudaGetLastError();
        START_CHECK_CUDA_ERROR(err, "Failed to launch particle dynamics kernel");

        // Wait for kernel to finish
        err = cudaDeviceSynchronize();
        START_CHECK_CUDA_ERROR(err, "Failed to synchronize after particle dynamics kernel");

        // 5. Prepare for surface collision detection
        // 5.1 Convert triangles to device format
        std::vector<DeviceTriangle> deviceTriangles;
        auto const& triangles = surfaceMesh.getTriangles();
        deviceTriangles.reserve(triangles.size());
        
        for (auto const& triangle : triangles)
        {
            DevicePoint v0(triangle.vertex(0).x(), triangle.vertex(0).y(), triangle.vertex(0).z());
            DevicePoint v1(triangle.vertex(1).x(), triangle.vertex(1).y(), triangle.vertex(1).z());
            DevicePoint v2(triangle.vertex(2).x(), triangle.vertex(2).y(), triangle.vertex(2).z());
            deviceTriangles.push_back(DeviceTriangle(v0, v1, v2));
        }
        
        // 5.2 Initialize the AABB tree for surface collision detection
        ParticleSurfaceCollisionHandlerCUDA::initialize(deviceTriangles);
        
        // 5.3 Allocate device memory for previous positions
        DevicePoint* d_previousPositions;
        err = cudaMalloc(&d_previousPositions, particles.size() * sizeof(DevicePoint));
        START_CHECK_CUDA_ERROR(err, "Failed to allocate device memory for previous positions");
        
        err = cudaMemcpy(d_previousPositions, previousPositions.data(), particles.size() * sizeof(DevicePoint), cudaMemcpyHostToDevice);
        START_CHECK_CUDA_ERROR(err, "Failed to copy previous positions to device");
        
        // 5.4 Allocate device memory for settled particles
        int* d_settledParticles;
        err = cudaMalloc(&d_settledParticles, particles.size() * sizeof(int));
        START_CHECK_CUDA_ERROR(err, "Failed to allocate device memory for settled particles");
        
        // Initialize settled particles array
        std::vector<int> settledParticlesArray(particles.size(), 0);
        for (auto const& id : settledParticlesIds)
        {
            for (size_t i = 0; i < particles.size(); ++i)
            {
                if (particles[i].getId() == id)
                {
                    settledParticlesArray[i] = 1;
                    break;
                }
            }
        }
        
        err = cudaMemcpy(d_settledParticles, settledParticlesArray.data(), particles.size() * sizeof(int), cudaMemcpyHostToDevice);
        START_CHECK_CUDA_ERROR(err, "Failed to copy settled particles to device");
        
        // 5.5 Allocate device memory for collision results
        DevicePoint* d_collisionPoints;
        int* d_collisionTriangles;
        int* d_numCollisions;
        
        err = cudaMalloc(&d_collisionPoints, particles.size() * sizeof(DevicePoint));
        START_CHECK_CUDA_ERROR(err, "Failed to allocate device memory for collision points");
        
        err = cudaMalloc(&d_collisionTriangles, particles.size() * sizeof(int));
        START_CHECK_CUDA_ERROR(err, "Failed to allocate device memory for collision triangles");
        
        err = cudaMalloc(&d_numCollisions, sizeof(int));
        START_CHECK_CUDA_ERROR(err, "Failed to allocate device memory for collision counter");
        
        // 6. Process surface collisions on GPU
        ParticleSurfaceCollisionHandlerCUDA::processCollisions(
            deviceParticles.get(),
            particles.size(),
            d_previousPositions,
            d_settledParticles,
            d_collisionPoints,
            d_collisionTriangles,
            d_numCollisions
        );
        
        // 7. Retrieve collision results
        int numCollisions = 0;
        err = cudaMemcpy(&numCollisions, d_numCollisions, sizeof(int), cudaMemcpyDeviceToHost);
        START_CHECK_CUDA_ERROR(err, "Failed to copy collision count from device");
        
        if (numCollisions > 0)
        {
            // 7.1 Copy collision data back to host
            std::vector<DevicePoint> collisionPoints(numCollisions);
            std::vector<int> collisionTriangles(numCollisions);
            
            err = cudaMemcpy(collisionPoints.data(), d_collisionPoints, numCollisions * sizeof(DevicePoint), cudaMemcpyDeviceToHost);
            START_CHECK_CUDA_ERROR(err, "Failed to copy collision points from device");
            
            err = cudaMemcpy(collisionTriangles.data(), d_collisionTriangles, numCollisions * sizeof(int), cudaMemcpyDeviceToHost);
            START_CHECK_CUDA_ERROR(err, "Failed to copy collision triangles from device");
            
            // 7.2 Update settled particles set and movement map
            std::vector<int> updatedSettledParticles(particles.size());
            err = cudaMemcpy(updatedSettledParticles.data(), d_settledParticles, particles.size() * sizeof(int), cudaMemcpyDeviceToHost);
            START_CHECK_CUDA_ERROR(err, "Failed to copy updated settled particles from device");
            
            for (size_t i = 0; i < particles.size(); ++i)
            {
                if (updatedSettledParticles[i] == 1 && settledParticlesArray[i] == 0)
                {
                    // This particle was settled during this step
                    size_t particleId = particles[i].getId();
                    
                    // Find the collision data for this particle
                    for (int j = 0; j < numCollisions; ++j)
                    {
                        // Add to settled particles set
                        {
                            std::unique_lock<std::shared_mutex> lock(sh_mutex_settledParticlesCounterMap);
                            settledParticlesIds.insert(particleId);
                        }
                        
                        // Update triangle counter in surface mesh
                        size_t triangleId = static_cast<size_t>(collisionTriangles[j]);
                        if (triangleId < surfaceMesh.getTriangleCellMap().size())
                        {
                            std::unique_lock<std::shared_mutex> lock(sh_mutex_settledParticlesCounterMap);
                            surfaceMesh.getTriangleCellMap()[triangleId].count += 1;
                            
                            // Check if we should stop the simulation
                            if (surfaceMesh.getTotalCountOfSettledParticles() >= particles.size())
                            {
                                stopSubject.notifyStopRequested();
                            }
                        }
                        
                        // Record movement
                        Point intersectionPoint(collisionPoints[j].x, collisionPoints[j].y, collisionPoints[j].z);
                        ParticleMovementTracker::recordMovement(
                            particleMovementMap,
                            mutex_particlesMovementMap,
                            particleId,
                            intersectionPoint
                        );
                        
                        break;
                    }
                }
            }
        }
        
        // 8. Free device memory
        cudaFree(d_previousPositions);
        cudaFree(d_settledParticles);
        cudaFree(d_collisionPoints);
        cudaFree(d_collisionTriangles);
        cudaFree(d_numCollisions);
        
        ParticleSurfaceCollisionHandlerCUDA::cleanup();

        // 9. Update host particles with results from GPU
        updateHostParticles(deviceParticles, particles);
        
        // Free device memory for particles
        deviceParticles.reset();
    }
    catch (std::exception const &e)
    {
        throw std::runtime_error(util::stringify("[ParticleDynamicsProcessorCUDA::process] ", e.what()));
    }
}

#endif // !USE_CUDA
