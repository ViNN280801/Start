#include "ParticleInCellEngine/ParticleDynamicsProcessor/ParticleMovementTracker.hpp"

ParticleMovementTracker::ParticleMovementTracker(ParticleMovementMap &movements,
                                                 std::mutex &mutex)
    : m_particlesMovement(movements),
      m_particlesMovementMutex(mutex) {}

void ParticleMovementTracker::recordMovement(Particle const &particle, Point const &position, size_t maxParticles)
{
    std::lock_guard<std::mutex> lock(m_particlesMovementMutex);
    if (m_particlesMovement.size() <= maxParticles)
        m_particlesMovement[particle.getId()].emplace_back(position);
}
