#ifndef PARTICLEDEVICE_CUH
#define PARTICLEDEVICE_CUH

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <vector>

/**
 * @brief Represents a single particle on the device.
 *
 * This structure contains information about the particle's ID, type,
 * position, velocity, and energy. It is designed for use in GPU computations.
 */
struct ParticleDevice_t
{
    size_t id;         ///< Unique identifier for the particle.
    int type;          ///< Particle type (e.g., int representation of `ParticleType`).
    double x, y, z;    ///< Position coordinates (x, y, z).
    double vx, vy, vz; ///< Velocity components (vx, vy, vz).
    double energy;     ///< Energy of the particle.
};

using ParticleDevice = ParticleDevice_t;
using ParticleDevice_ref = ParticleDevice_t &;
using ParticleDevice_rref = ParticleDevice_t &&;
using ParticleDevice_cref = ParticleDevice_t const &;


/**
 * @brief Manages a dynamic array of particles on the GPU.
 *
 * This class provides utilities for managing GPU memory for particle data,
 * copying data between host and device, and resizing GPU arrays.
 */
class ParticleDeviceArray
{
private:
    ParticleDevice_t *d_particles{}; ///< Pointer to particles on the device.
    size_t count{};                  ///< Number of particles.

public:
    /**
     * @brief Default constructor.
     */
    ParticleDeviceArray() = default;

    /**
     * @brief Constructor to initialize with existing device memory.
     * @param particles Pointer to device memory containing particles.
     * @param count Number of particles in the array.
     */
    ParticleDeviceArray(ParticleDevice_t *particles, size_t count);

    /**
     * @brief Destructor to free GPU memory.
     */
    ~ParticleDeviceArray();

    /**
     * @brief Move constructor.
     * @param other Source object to move from.
     */
    ParticleDeviceArray(ParticleDeviceArray &&other) noexcept;

    /**
     * @brief Move assignment operator.
     * @param other Source object to move from.
     * @return Reference to the current object.
     */
    ParticleDeviceArray &operator=(ParticleDeviceArray &&other) noexcept;

    // Deleted copy semantics to avoid accidental copies.
    ParticleDeviceArray(ParticleDeviceArray const &) = delete;
    ParticleDeviceArray &operator=(ParticleDeviceArray const &) = delete;

    /**
     * @brief Getter for the device particles ptr.
     */
    ParticleDevice_t *get() const { return d_particles; }

    /**
     * @brief Resets the ptr on device particle.
     * @details 1. Deallocates memory;
     *          2. Assigns ptr to `nullptr.
     */
    void reset();

    /**
     * @brief Check if the array is empty.
     * @return True if the array is empty, false otherwise.
     */
    bool empty() const;

    /**
     * @brief Getter for the number of particles.
     */
    size_t size() const { return count; }

    /**
     * @brief Get a pointer to the beginning of the device array.
     * @return Pointer to the first particle.
     */
    ParticleDevice_t *begin();

    /**
     * @brief Get a constant pointer to the beginning of the device array.
     * @return Constant pointer to the first particle.
     */
    ParticleDevice_t const *cbegin() const;

    /**
     * @brief Get a pointer to the end of the device array.
     * @return Pointer to the end of the particle array.
     */
    ParticleDevice_t *end();

    /**
     * @brief Get a constant pointer to the end of the device array.
     * @return Constant pointer to the end of the particle array.
     */
    ParticleDevice_t const *cend() const;

    /**
     * @brief Resize the particle array on the device.
     * @param newCount New number of particles.
     */
    void resize(size_t newCount);

    /**
     * @brief Access particle at index without bounds checking.
     * @param index Index of particle to access.
     * @return Reference to the particle at the given index.
     */
    ParticleDevice_t &operator[](size_t index);
    
    /**
     * @brief Access particle at index without bounds checking (const version).
     * @param index Index of particle to access.
     * @return Const reference to the particle at the given index.
     */
    ParticleDevice_t const &operator[](size_t index) const;

    /**
     * @brief Access particle at index with bounds checking.
     * @param index Index of particle to access.
     * @return Reference to the particle at the given index.
     * @throws std::out_of_range if index is out of bounds.
     */
    ParticleDevice_t &at(size_t index);
    
    /**
     * @brief Access particle at index with bounds checking (const version).
     * @param index Index of particle to access.
     * @return Const reference to the particle at the given index.
     * @throws std::out_of_range if index is out of bounds.
     */
    ParticleDevice_t const &at(size_t index) const;
};

#endif // !USE_CUDA

#endif // !PARTICLEDEVICE_CUH
