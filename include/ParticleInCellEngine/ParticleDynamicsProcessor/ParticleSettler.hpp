#ifndef PARTICLESETTLER_HPP
#define PARTICLESETTLER_HPP

#include <shared_mutex>

#include "ParticleInCellEngine/PICTypes.hpp"
#include "ParticleInCellEngine/ParticleDynamicsProcessor/StopModelingObserver.hpp"

/**
 * @class ParticleSettler
 * @brief Handles settling particles on mesh surfaces and maintains particle settlement state.
 *
 * This class provides functionality to check if a particle is already settled
 * and to update settlement information when a particle collides with a surface.
 * It ensures thread safety using a shared mutex.
 */
class ParticleSettler : public StopObserver
{
private:
    std::shared_mutex &m_settledParticlesMutex;               ///< Mutex to protect access to settled particles.
    ParticlesIDSet &m_settledParticleIds;                     ///< Set of IDs of particles that have settled.
    SettledParticlesCounterMap &m_settledParticlesCounterMap; ///< Map of triangle IDs to particle counts.
    StopSubject &m_subject;                                   ///< Subject reference to manage stop modeling flag.

public:
    /**
     * @brief Constructs a ParticleSettler object.
     * @param mutex Reference to the shared mutex for thread-safe operations.
     * @param ids Reference to the set of settled particle IDs.
     * @param counterMap Reference to the map of triangle IDs and their settled particle counts.
     */
    ParticleSettler(std::shared_mutex &mutex,
                    ParticlesIDSet &ids,
                    SettledParticlesCounterMap &counterMap,
                    StopSubject &subject);

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

    /**
     * @brief Implementation of StopObserver's onStopRequested.
     * In this case, ParticleSettler not need special action, but it's here for correct compiling.
     */
    void onStopRequested() override {}
};

#endif // !PARTICLESETTLER_HPP
