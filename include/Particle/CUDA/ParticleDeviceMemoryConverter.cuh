#ifndef PARTICLEDEVICEMEMORYCONVERTER_CUH
#define PARTICLEDEVICEMEMORYCONVERTER_CUH

#ifdef USE_CUDA

#include <vector>

#include "Particle/ParticleDevice.cuh"

/**
 * @brief Handles memory conversions for `ParticleDevice_t` and `ParticleDeviceArray`.
 *
 * This class provides utility methods to allocate, deallocate, and copy data
 * for single particles and arrays of particles between host and device memory.
 */
class ParticleDeviceMemoryConverter
{
public:
    /**
     * @brief Allocates memory on the GPU for a single `ParticleDevice_t`.
     * @param particle A reference to the particle on the host.
     * @return A pointer to the particle on the device.
     */
    static ParticleDevice_t *allocateDeviceMemory(ParticleDevice_t const &particle);

    /**
     * @brief Allocates memory on the GPU for an array of `ParticleDevice_t`.
     * @param particles A vector of particles on the host.
     * @return A `ParticleDeviceArray` containing the device data.
     */
    static ParticleDeviceArray allocateDeviceArrayMemory(std::vector<ParticleDevice_t> const &particles);

    /**
     * @brief Copies a single `ParticleDevice_t` from the device to the host.
     * @param d_particle Pointer to the particle on the device.
     * @return The copied particle on the host.
     */
    static ParticleDevice_t copyToHost(ParticleDevice_t const *d_particle);

    /**
     * @brief Copies a `ParticleDeviceArray` from the device to the host.
     * @param deviceArray The particle array on the device.
     * @return A vector of particles on the host.
     */
    static std::vector<ParticleDevice_t> copyToHost(ParticleDeviceArray const &deviceArray);

    /**
     * @brief Frees GPU memory for a single `ParticleDevice_t`.
     * @param d_particle Pointer to the particle on the device.
     */
    static void freeDeviceMemory(ParticleDevice_t *d_particle);

    /**
     * @brief Frees GPU memory for a `ParticleDeviceArray`.
     * @param deviceArray The particle array on the device.
     */
    static void freeDeviceArrayMemory(ParticleDeviceArray &deviceArray);
};

#endif // !USE_CUDA

#endif // !PARTICLEDEVICEMEMORYCONVERTER_CUH
