#include "ParticleInCellEngine/ParticleDynamicsProcessor/ParticleMovementTracker.hpp"

ParticleMovementTracker::ParticleMovementTracker(ParticleMovementMap &movements,
												 std::mutex &mutex)
	: m_particlesMovement(movements),
	  m_particlesMovementMutex(mutex) {}

void ParticleMovementTracker::recordMovement(size_t particleId, Point const &position, size_t maxParticles) noexcept
{
	std::lock_guard<std::mutex> lock(m_particlesMovementMutex);
	if (m_particlesMovement.size() <= maxParticles)
		m_particlesMovement[particleId].emplace_back(position);
}
