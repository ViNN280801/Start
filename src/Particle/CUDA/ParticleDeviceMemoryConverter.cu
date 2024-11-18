#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <stdexcept>

#include "Particle/ParticleDeviceMemoryConverter.cuh"

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
    return deviceArray.copyToHost();
}

void ParticleDeviceMemoryConverter::freeDeviceMemory(ParticleDevice_t *d_particle)
{
    if (d_particle)
    {
        cudaFree(d_particle);
        d_particle = nullptr;
    }
}

void ParticleDeviceMemoryConverter::freeDeviceArrayMemory(ParticleDeviceArray &deviceArray)
{
    if (deviceArray.d_particles)
    {
        cudaFree(deviceArray.d_particles);
        deviceArray.d_particles = nullptr;
    }
    deviceArray.count = 0;
}

#endif // USE_CUDA
