#include <cuda_runtime.h>
#include <stdexcept>

#include "Particle/CUDA/ParticleDeviceMemoryConverter.cuh"
#include "Utilities/CUDA/DeviceUtils.cuh"

ParticleDevice_t *ParticleDeviceMemoryConverter::allocateDeviceMemory(ParticleDevice_t const &particle)
{
    ParticleDevice_t *d_particle = nullptr;
    cudaMalloc(&d_particle, sizeof(ParticleDevice_t));
    cudaMemcpy(d_particle, &particle, sizeof(ParticleDevice_t), cudaMemcpyHostToDevice);
    return d_particle;
}

ParticleDeviceArray ParticleDeviceMemoryConverter::allocateDeviceArrayMemory(std::vector<ParticleDevice_t> const &particles)
{
    ParticleDeviceArray deviceArray;
    deviceArray.resize(particles.size());
    cudaMemcpy(deviceArray.begin(), particles.data(), particles.size() * sizeof(ParticleDevice_t), cudaMemcpyHostToDevice);
    return deviceArray;
}

Particle ParticleDeviceMemoryConverter::copyToHost(ParticleDevice_t const *d_particle)
{
    if (!d_particle)
        throw std::runtime_error("Device particle pointer is null");
    ParticleDevice_t hostDeviceParticle;
    cuda_utils::check_cuda_err(cudaMemcpy(&hostDeviceParticle, d_particle, sizeof(ParticleDevice_t), cudaMemcpyDeviceToHost),
                               "CUDA memcpy device to host failed");
    Particle particle(hostDeviceParticle);
    return particle;
}

ParticleVector ParticleDeviceMemoryConverter::copyToHost(ParticleDeviceArray const &deviceArray)
{
    // Check if the device array is empty
    if (deviceArray.empty())
        return {}; // Return an empty vector if no particles exist

    if (!deviceArray.get())
        throw std::runtime_error("Device particle array pointer is null");

    START_CHECK_CUDA_ERROR(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed during copyToHost");

    // Allocate host memory to hold the particles
    std::vector<ParticleDevice_t> hostDeviceParticles(deviceArray.size());

    // Copy data from the device to the host with enhanced error checking
    START_CHECK_CUDA_ERROR(cudaMemcpy(
                               hostDeviceParticles.data(),                    // Host destination
                               deviceArray.get(),                             // Device source
                               deviceArray.size() * sizeof(ParticleDevice_t), // Size of data in bytes
                               cudaMemcpyDeviceToHost),                       // Direction: device -> host
                           "CUDA memcpy device to host failed");

    // Verify successful copy by checking cudaGetLastError
    START_CHECK_CUDA_ERROR(cudaGetLastError(), "Error after cudaMemcpy in copyToHost");

    ParticleVector particles;
    particles.reserve(hostDeviceParticles.size());

    // Use a try-catch block to identify which particle causes problems
    for (size_t i = 0; i < hostDeviceParticles.size(); ++i)
    {
        try
        {
            particles.emplace_back(hostDeviceParticles[i]);
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error(
                "Failed to convert device particle " + std::to_string(i) +
                " to host particle: " + e.what());
        }
    }

    return particles;
}

void ParticleDeviceMemoryConverter::freeDeviceMemory(ParticleDevice_t *d_particle)
{
    if (d_particle)
    {
        cudaFree(d_particle);
        d_particle = nullptr;
    }
}

void ParticleDeviceMemoryConverter::freeDeviceArrayMemory(ParticleDeviceArray &deviceArray) { deviceArray.reset(); }
