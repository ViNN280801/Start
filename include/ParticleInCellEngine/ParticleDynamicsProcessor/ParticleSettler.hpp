#ifndef PARTICLE_SETTLER_HPP
#define PARTICLE_SETTLER_HPP

#include <shared_mutex>

#include "Geometry/Mesh/Surface/SurfaceMesh.hpp"
#include "ParticleInCellEngine/PICTypes.hpp"
#include "ParticleInCellEngine/ParticleDynamicsProcessor/StopModelingObserver.hpp"

/**
 * @class ParticleSettler
 * @brief Manages the process of settling particles on a surface mesh and maintains their settlement state.
 *
 * The `ParticleSettler` class provides mechanisms for detecting and marking settled particles
 * in a particle-in-cell (PIC) simulation. It ensures thread safety while modifying shared data structures.
 *
 * @details
 * - Implements **thread-safe particle tracking** with `std::shared_mutex` to allow concurrent access.
 * - Notifies a **`StopSubject`** when all particles are settled, ensuring proper termination of the simulation.
 * - Maintains a **set of settled particle IDs** to avoid redundant processing.
 * - Increments the **particle count** for each triangle in the mesh when a particle settles on it.
 *
 * @note This class is derived from `StopObserver` to allow proper integration with the simulation's stop mechanism.
 *
 * @thread_safety
 * - Read operations use `std::shared_lock` for concurrent access.
 * - Write operations use `std::unique_lock` to ensure exclusive access.
 *
 * **Key Responsibilities:**
 * - Determines if a particle is already settled (`isSettled`).
 * - Updates the particle settlement state (`settle`).
 * - Ensures proper **simulation termination** when all particles are settled.
 *
 * **Usage Example:**
 * @code
 * ParticlesIDSet settledParticlesIds;
 * std::shared_mutex sh_mutex;
 * SurfaceMesh surfaceMesh("TwoPlates.msh");
 * StopSubject stopSubject;
 *
 * size_t particleId = 42;
 * size_t triangleId = 10;
 * size_t totalParticles = 10000;
 *
 * if (!ParticleSettler::isSettled(particleId, settledParticlesIds, sh_mutex))
 * {
 *     ParticleSettler::settle(particleId, triangleId, totalParticles, surfaceMesh, sh_mutex, settledParticlesIds, stopSubject);
 * }
 * @encode
 */
class ParticleSettler : public StopObserver
{
public:
    /**
     * @brief Checks if a particle has already settled on the surface.
     *
     * This method determines whether a given particle ID is already present in the
     * `settledParticlesIds` set, indicating that it has previously settled.
     *
     * @param[in] particleId The unique identifier of the particle being checked.
     * @param[in] settledParticlesIds The set of settled particle IDs.
     * @param[in,out] sh_mutex_settledParticlesCounterMap A shared mutex ensuring thread-safe access to the settled particle set.
     *
     * @return `true` if the particle is already settled, `false` otherwise.
     *
     * @thread_safety
     * - Uses `std::shared_lock<std::shared_mutex>` to allow multiple concurrent read operations.
     *
     * **Example Usage:**
     * @code
     * if (ParticleSettler::isSettled(particleId, settledParticlesIds, sh_mutex))
     * {
     *     std::cout << "Particle " << particleId << " is already settled." << std::endl;
     * }
     * @endcode
     *
     * @warning This method only checks for presence in `settledParticlesIds` and does not verify if the triangle itself is full.
     */
    static bool isSettled(size_t particleId,
                          ParticlesIDSet_cref settledParticlesIds,
                          std::shared_mutex &sh_mutex_settledParticlesCounterMap);

    /**
     * @brief Marks a particle as settled and updates settlement data structures.
     *
     * This method updates the surface mesh to reflect the settling of a particle.
     * It increments the particle count on the corresponding triangle, adds the particle ID
     * to the set of settled particles, and checks if the total number of settled particles
     * meets or exceeds the total particle count, signaling termination if necessary.
     *
     * @param[in] particleId The unique identifier of the particle being settled.
     * @param[in] triangleId The ID of the triangle where the particle has settled.
     * @param[in] totalParticles The total number of particles in the simulation.
     * @param[in,out] surfaceMesh The `SurfaceMesh` object managing the simulation's surface representation.
     * @param[in,out] sh_mutex_settledParticlesCounterMap A shared mutex ensuring thread-safe updates to the settlement state.
     * @param[in,out] settledParticlesIds A set containing IDs of all settled particles.
     * @param[in,out] stopSubject A reference to the `StopSubject`, which is notified if all particles are settled.
     *
     * @details
     * **Algorithm:**
     * 1. Acquire a `std::unique_lock<std::shared_mutex>` to ensure **exclusive** access to settlement data.
     * 2. Increment the **settled particle count** for the corresponding triangle in the mesh.
     * 3. Insert the **particle ID** into the `settledParticlesIds` set to prevent redundant processing.
     * 4. Check if the **total settled particle count** has reached the total particle count.
     *    - If all particles are settled, notify `stopSubject` to terminate the simulation.
     *
     * **Example Usage:**
     * @code
     * ParticleSettler::settle(particleId, triangleId, totalParticles, surfaceMesh, sh_mutex, settledParticlesIds, stopSubject);
     * @endcode
     *
     * @warning This method ensures **thread-safe** updates but must be called within an appropriate simulation loop.
     */
    static void settle(size_t particleId,
                       size_t triangleId,
                       size_t totalParticles,
                       SurfaceMesh_ref surfaceMesh,
                       std::shared_mutex &sh_mutex_settledParticlesCounterMap,
                       ParticlesIDSet_ref settledParticlesIds,
                       StopSubject &stopSubject);

    /**
     * @brief Implementation of StopObserver's onStopRequested.
     * In this case, ParticleSettler not need special action, but it's here for correct compiling.
     */
    void onStopRequested() override {}
};

#endif // !PARTICLE_SETTLER_HPP
