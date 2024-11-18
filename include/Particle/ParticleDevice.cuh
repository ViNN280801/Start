#ifndef PARTICLEDEVICE_CUH
#define PARTICLEDEVICE_CUH

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <stddef.h>

#include "Particle/ParticleDevice.cuh"

struct ParticleDevice_t
{
    size_t id;
    int type; // Using `int` for `ParticleType`
    double x, y, z;
    double vx, vy, vz;
    double energy;
};

struct ParticleDeviceArray_t
{
    ParticleDevice_t *d_particles = nullptr;
    size_t count = 0ul;

    ParticleDeviceArray_t() = default;
    ParticleDeviceArray_t(ParticleDevice_t *particles, size_t count) : d_particles(particles), count(count) {}
    ~ParticleDeviceArray_t()
    {
        if (d_particles)
        {
            cudaFree(d_particles);
            d_particles = nullptr;
        }
    }

    ParticleDeviceArray_t(const ParticleDeviceArray_t &) = delete;
    ParticleDeviceArray_t &operator=(const ParticleDeviceArray_t &) = delete;

    ParticleDeviceArray_t(ParticleDeviceArray_t &&other) noexcept
    {
        d_particles = other.d_particles;
        count = other.count;
        other.d_particles = nullptr;
        other.count = 0;
    }

    ParticleDeviceArray_t &operator=(ParticleDeviceArray_t &&other) noexcept
    {
        if (this != &other)
        {
            if (d_particles)
                cudaFree(d_particles);
            d_particles = other.d_particles;
            count = other.count;
            other.d_particles = nullptr;
            other.count = 0;
        }
        return *this;
    }

    bool empty() const { return count == 0; }
    ParticleDevice_t *begin() { return d_particles; }
    ParticleDevice_t const *begin() const { return d_particles; }
    ParticleDevice_t *end() { return d_particles + count; }
    ParticleDevice_t const *end() const { return d_particles + count; }

    void copyFromHost(std::vector<ParticleDevice_t> const &hostParticles)
    {
        if (hostParticles.size() != count)
            throw std::runtime_error("Size mismatch during host-to-device copy");
        cudaMemcpy(d_particles, hostParticles.data(), count * sizeof(ParticleDevice_t), cudaMemcpyHostToDevice);
    }
    std::vector<ParticleDevice_t> copyToHost() const
    {
        std::vector<ParticleDevice_t> hostParticles(count);
        cudaMemcpy(hostParticles.data(), d_particles, count * sizeof(ParticleDevice_t), cudaMemcpyDeviceToHost);
        return hostParticles;
    }

    void resize(size_t newCount)
    {
        if (d_particles)
            cudaFree(d_particles);
        count = newCount;
        cudaMalloc(&d_particles, count * sizeof(ParticleDevice_t));
    }
};

#endif // !USE_CUDA

#endif // !PARTICLEDEVICE_CUH
