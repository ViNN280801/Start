#ifndef PARTICLE_MOVEMENT_TRACKER_HPP
#define PARTICLE_MOVEMENT_TRACKER_HPP

#include "Particle/Particle.hpp"
#include "ParticleInCellEngine/PICTypes.hpp"

/**
 * @class ParticleMovementTracker
 * @brief Tracks and records the movement of particles in a Particle-In-Cell (PIC) simulation.
 *
 * The `ParticleMovementTracker` class is responsible for storing the trajectories of particles
 * as they move through the simulation space. It maintains a thread-safe structure to record
 * particle movements for further visualization and analysis.
 *
 * @details
 * - Uses a **thread-safe mechanism** via `std::mutex` to ensure proper synchronization when updating particle positions.
 * - Limits the number of recorded particles to avoid excessive memory usage.
 * - Designed for **high-performance simulations**, allowing for concurrent updates to the particle movement map.
 *
 * @note This class only provides a static method and is not intended to be instantiated.
 *
 * @thread_safety
 * - Uses `std::mutex` to protect concurrent access to `ParticleMovementMap`.
 *
 * **Key Responsibilities:**
 * - Tracks particle movement throughout the simulation.
 * - Stores the trajectory of each particle in `ParticleMovementMap`.
 * - Prevents memory overload by limiting the number of tracked particles.
 *
 * **Usage Example:**
 * @code
 * ParticleMovementMap movementMap;
 * std::mutex movementMutex;
 * size_t particleId = 42;
 * Point newPosition(10.0, 5.0, 3.0);
 *
 * ParticleMovementTracker::recordMovement(movementMap, movementMutex, particleId, newPosition);
 * @endcode
 */
class ParticleMovementTracker
{
private:
    constexpr static size_t kdefault_max_particles_to_record{200'000ul}; ///< Default number of the particles to record them.

public:
    /**
     * @brief Records the movement of a particle at a given time step.
     *
     * This method updates the `ParticleMovementMap` with the new position of a particle,
     * ensuring thread safety by locking access with `std::mutex`. The movement is recorded
     * only if the total number of stored particles does not exceed `maxParticles`.
     *
     * @param[in,out] particlesMovementMap A map storing particle movements, where each key represents
     *                                     a particle ID and the value is a list of recorded positions.
     * @param[in,out] mutex_particlesMovement A mutex ensuring safe concurrent access to `particlesMovement`.
     * @param[in] particleId The unique identifier of the particle being tracked.
     * @param[in] position The current position of the particle in 3D space.
     * @param[in] maxParticles The maximum number of particles that can be recorded. Defaults to `kdefault_max_particles_to_record`.
     *
     * @details
     * - The function ensures that **only a limited number of particles** are recorded to prevent excessive memory usage.
     * - If `particlesMovement.size()` exceeds `maxParticles`, no further movement is recorded for additional particles.
     * - Uses **a thread-safe mechanism** with `std::lock_guard<std::mutex>` to prevent race conditions.
     *
     * @return void
     *
     * @exception noexcept
     * This function is marked `noexcept` as it does not throw exceptions.
     *
     * **Algorithm:**
     * 1. **Acquire lock** on `particlesMovement` using `std::lock_guard<std::mutex>`.
     * 2. **Check particle limit**: If `particlesMovement.size()` exceeds `maxParticles`, stop recording.
     * 3. **Update movement map**: Append `position` to the trajectory list for `particleId`.
     *
     * **Example Usage:**
     * @code
     * ParticleMovementMap movementMap;
     * std::mutex movementMutex;
     * size_t particleId = 101;
     * Point newPosition(25.4, 12.8, -4.3);
     *
     * ParticleMovementTracker::recordMovement(movementMap, movementMutex, particleId, newPosition);
     * @endcode
     *
     * @warning **Memory Efficiency Notice**:
     * If too many particles are recorded, memory consumption can become excessive. The function
     * prevents this by enforcing a maximum number of tracked particles.
     */
    static void recordMovement(ParticleMovementMap &particlesMovementMap,
                               std::mutex &mutex_particlesMovement,
                               size_t particleId,
                               Point const &position,
                               size_t maxParticles = kdefault_max_particles_to_record) noexcept;

    /**
     * @brief Saves the particle movements to a JSON file.
     *
     * This function saves the contents of `particlesMovementMap` to a JSON file named `filepath`.
     * It handles exceptions and provides a warning message if the map is empty.
     */
    static void saveMovementsToJson(ParticleMovementMap const &particlesMovementMap,
                                    std::string_view filepath = "results/particles_movements.json");
};

#endif // !PARTICLE_MOVEMENT_TRACKER_HPP
