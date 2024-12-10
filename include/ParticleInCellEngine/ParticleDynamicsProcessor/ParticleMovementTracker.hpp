#ifndef PARTICLEMOVEMENTTRACKER_HPP
#define PARTICLEMOVEMENTTRACKER_HPP

#include "Particle/Particle.hpp"
#include "ParticleInCellEngine/PICTypes.hpp"

/**
 * @class ParticleMovementTracker
 * @brief Tracks the movement of particles and records their trajectories.
 *
 * This class ensures thread-safe recording of particle movements and maintains
 * a map of particle trajectories for visualization and analysis.
 */
class ParticleMovementTracker
{
private:
    ParticleMovementMap &m_particlesMovement; ///< Map storing particle movements.
    std::mutex &m_particlesMovementMutex;     ///< Mutex to protect access to the movement map.

    constexpr static size_t kdefault_max_particles_to_record{5'000ul}; ///< Default number of the particles to record them.

public:
    /**
     * @brief Constructs a ParticleMovementTracker object.
     * @param movements Reference to the map storing particle movements.
     * @param mutex Reference to the mutex for thread-safe operations.
     */
    ParticleMovementTracker(ParticleMovementMap &movements, std::mutex &mutex);

    /**
     * @brief Records the movement of a particle.
     * @param particle The particle being tracked.
     * @param position The current position of the particle.
     * @param maxParticles Maximum number of particles to track.
     */
    void recordMovement(Particle const &particle, Point const &position, size_t maxParticles = kdefault_max_particles_to_record) noexcept;
};

#endif // PARTICLEMOVEMENTTRACKER_HPP
