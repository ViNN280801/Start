#ifndef PARTICLESETTLER_HPP
#define PARTICLESETTLER_HPP

#include <shared_mutex>

#include "ParticleInCellEngine/PICTypes.hpp"

/**
 * @class ParticleSettler
 * @brief Handles settling particles on mesh surfaces and maintains particle settlement state.
 *
 * This class provides functionality to check if a particle is already settled
 * and to update settlement information when a particle collides with a surface.
 * It ensures thread safety using a shared mutex.
 */
class ParticleSettler
{
private:
    std::shared_mutex &m_settledParticlesMutex;               ///< Mutex to protect access to settled particles.
    ParticlesIDSet &m_settledParticleIds;                     ///< Set of IDs of particles that have settled.
    SettledParticlesCounterMap &m_settledParticlesCounterMap; ///< Map of triangle IDs to particle counts.

public:
    /**
     * @brief Constructs a ParticleSettler object.
     * @param mutex Reference to the shared mutex for thread-safe operations.
     * @param ids Reference to the set of settled particle IDs.
     * @param counterMap Reference to the map of triangle IDs and their settled particle counts.
     */
    ParticleSettler(std::shared_mutex &mutex,
                    ParticlesIDSet &ids,
                    SettledParticlesCounterMap &counterMap);

    /**
     * @brief Checks if a particle is already settled.
     * @param particleId The ID of the particle to check.
     * @return `true` if the particle is settled, otherwise `false`.
     */
    bool isSettled(size_t particleId);

    /**
     * @brief Settles a particle on a specific triangle.
     * @param particleId The ID of the particle to settle.
     * @param triangleId The ID of the triangle where the particle settles.
     * @param totalParticles Total number of particles in the simulation.
     * @throws std::runtime_error If all particles are settled.
     */
    void settle(size_t particleId, size_t triangleId, size_t totalParticles);
};

#endif // !PARTICLESETTLER_HPP
