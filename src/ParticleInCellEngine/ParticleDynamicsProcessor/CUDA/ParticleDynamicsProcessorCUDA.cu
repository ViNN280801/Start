#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <vector>

#include "Particle/CUDA/ParticleDevice.cuh"
#include "Particle/CUDA/ParticleDeviceMemoryConverter.cuh"
#include "ParticleInCellEngine/ParticleDynamicsProcessor/CUDA/ParticleDynamicsProcessorCUDA.cuh"
#include "ParticleInCellEngine/ParticleDynamicsProcessor/CUDA/ParticleKernelHelpers.cuh"
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
__global__ void particleDynamicsKernel(ParticleDevice_t *particles, size_t count, ParticleDynamicsKernelParams params)
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
            hostParticles[i].setVelocity(
                updatedParticles[i].getVx(),
                updatedParticles[i].getVy(),
                updatedParticles[i].getVz());
            hostParticles[i].setEnergy_J(updatedParticles[i].getEnergy_J());
            // The position is already updated by the kernel
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
        // Skip if there are no particles to process
        if (particles.empty())
            return;

        // 1. Prepare particles for GPU processing
        auto deviceParticles = prepareParticlesForGPU(particles);

        // 2. Record previous positions for movement tracking (need to do this before sending to GPU)
        std::vector<Point> previousPositions;
        previousPositions.reserve(particles.size());
        for (auto &particle : particles)
        {
            previousPositions.push_back(particle.getCentre());
        }

        // 3. Prepare kernel parameters
        ParticleDynamicsKernelParams params;
        params.timeMoment = timeMoment;
        params.timeStep = timeStep;
        params.gasConcentration = gasConcentration;
        params.scatteringModelType = convertScatteringModelToInt(scatteringModel);
        params.gasType = convertGasNameToParticleType(gasName);

        // 4. Launch kernel
        int numThreads = MAX_THREADS_PER_BLOCK;
        int numBlocks = static_cast<int>((particles.size() + numThreads - 1) / numThreads);

        particleDynamicsKernel<<<numBlocks, numThreads>>>(
            deviceParticles.get(),
            particles.size(),
            params);

        // Check for kernel errors
        cudaError_t err = cudaGetLastError();
        cuda_utils::check_cuda_err(err, "Failed to launch particle dynamics kernel");

        // Wait for kernel to finish
        err = cudaDeviceSynchronize();
        cuda_utils::check_cuda_err(err, "Failed to synchronize after particle dynamics kernel");

        // 5. Update host particles with results from GPU
        updateHostParticles(deviceParticles, particles);

        // 6. Process particle movement and surface collisions on CPU
        // These operations involve complex data structures not easily handled on GPU
        for (size_t i = 0; i < particles.size(); ++i)
        {
            auto &particle = particles[i];

            // Skip already settled particles
            if (ParticleSettler::isSettled(particle.getId(), settledParticlesIds, sh_mutex_settledParticlesCounterMap))
                continue;

            // Record particle movement
            ParticleMovementTracker::recordMovement(
                particleMovementMap,
                mutex_particlesMovementMap,
                particle.getId(),
                previousPositions[i]);

            // Create ray from previous to current position for collision detection
            Ray ray(previousPositions[i], particle.getCentre());
            if (!ray.is_degenerate())
            {
                // Handle surface collisions
                auto collision = ParticleSurfaceCollisionHandler::handle(
                    particle,
                    ray,
                    particles.size(),
                    surfaceMesh,
                    sh_mutex_settledParticlesCounterMap,
                    mutex_particlesMovementMap,
                    particleMovementMap,
                    settledParticlesIds,
                    stopSubject);
            }
        }

        // Free device memory
        deviceParticles.reset();
    }
    catch (std::exception const &e)
    {
        throw std::runtime_error(util::stringify("[ParticleDynamicsProcessorCUDA::process] ", e.what()));
    }
}

#endif // USE_CUDA