#include <cassert>
#include <cuda_runtime.h>
#include <stdexcept>

#include "Particle/CUDA/ParticleDevice.cuh"

ParticleDeviceArray::ParticleDeviceArray(ParticleDevice_t *particles, size_t count)
    : d_particles(particles), count(count) {}

ParticleDeviceArray::~ParticleDeviceArray() { reset(); }

ParticleDeviceArray::ParticleDeviceArray(ParticleDeviceArray &&other) noexcept
    : d_particles(other.d_particles), count(other.count)
{
    other.d_particles = nullptr;
    other.count = 0;
}

ParticleDeviceArray &ParticleDeviceArray::operator=(ParticleDeviceArray &&other) noexcept
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

void ParticleDeviceArray::reset()
{
    if (d_particles)
    {
        cudaFree(d_particles);
        d_particles = nullptr;
    }
    count = 0ul;
}

bool ParticleDeviceArray::empty() const { return count == 0; }
ParticleDevice_t *ParticleDeviceArray::begin() { return d_particles; }
ParticleDevice_t const *ParticleDeviceArray::cbegin() const { return d_particles; }
ParticleDevice_t *ParticleDeviceArray::end() { return d_particles + count; }
ParticleDevice_t const *ParticleDeviceArray::cend() const { return d_particles + count; }

void ParticleDeviceArray::resize(size_t newCount)
{
    if (d_particles)
        cudaFree(d_particles);

    count = newCount;
    cudaMalloc(&d_particles, count * sizeof(ParticleDevice_t));
}

ParticleDevice_t &ParticleDeviceArray::operator[](size_t index)
{
    assert(index < count && "Index out of bounds");
    return d_particles[index];
}

ParticleDevice_t const &ParticleDeviceArray::operator[](size_t index) const
{
    assert(index < count && "Index out of bounds");
    return d_particles[index];
}

ParticleDevice_t &ParticleDeviceArray::at(size_t index)
{
    if (index >= count)
        throw std::out_of_range("ParticleDeviceArray::at: index out of range");
    return d_particles[index];
}

ParticleDevice_t const &ParticleDeviceArray::at(size_t index) const
{
    if (index >= count)
        throw std::out_of_range("ParticleDeviceArray::at: index out of range");
    return d_particles[index];
}
