#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <stdexcept>

#include "Particle/CUDA/ParticleDeviceMemoryConverter.cuh"

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

ParticleDevice_t ParticleDeviceMemoryConverter::copyToHost(ParticleDevice_t const *d_particle)
{
    if (!d_particle)
        throw std::runtime_error("Device particle pointer is null");
    ParticleDevice_t particle;
    cudaMemcpy(&particle, d_particle, sizeof(ParticleDevice_t), cudaMemcpyDeviceToHost);
    return particle;
}

std::vector<ParticleDevice_t> ParticleDeviceMemoryConverter::copyToHost(ParticleDeviceArray const &deviceArray)
{
    // Check if the device array is empty
    if (deviceArray.empty())
    {
        return {}; // Return an empty vector if no particles exist
    }

    // Allocate host memory to hold the particles
    std::vector<ParticleDevice_t> hostParticles(deviceArray.size());

    // Copy data from the device to the host
    cudaError_t err = cudaMemcpy(
        hostParticles.data(),                          // Host destination
        deviceArray.get(),                             // Device source
        deviceArray.size() * sizeof(ParticleDevice_t), // Size of data in bytes
        cudaMemcpyDeviceToHost);                       // Direction: device -> host

    // Check for CUDA errors
    if (err != cudaSuccess)
    {
        throw std::runtime_error(
            std::string("CUDA memcpy device to host failed: ") + cudaGetErrorString(err));
    }

    // Return the copied particles
    return hostParticles;
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

#endif // USE_CUDA
