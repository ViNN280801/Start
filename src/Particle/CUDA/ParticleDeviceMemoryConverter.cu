#ifdef USE_CUDA

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

    // Allocate host memory to hold the particles
    std::vector<ParticleDevice_t> hostDeviceParticles(deviceArray.size());

    // Copy data from the device to the host
    cuda_utils::check_cuda_err(cudaMemcpy(
                                   hostDeviceParticles.data(),                          // Host destination
                                   deviceArray.get(),                             // Device source
                                   deviceArray.size() * sizeof(ParticleDevice_t), // Size of data in bytes
                                   cudaMemcpyDeviceToHost),                       // Direction: device -> host
                               "CUDA memcpy device to host failed");
    ParticleVector particles(hostDeviceParticles.cbegin(), hostDeviceParticles.cend());
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

#endif // USE_CUDA
