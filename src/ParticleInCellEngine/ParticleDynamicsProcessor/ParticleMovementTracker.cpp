#include "ParticleInCellEngine/ParticleDynamicsProcessor/ParticleMovementTracker.hpp"

void ParticleMovementTracker::recordMovement(ParticleMovementMap &particlesMovementMap,
											 std::mutex &mutex_particlesMovement,
											 size_t particleId,
											 Point const &position,
											 size_t maxParticles) noexcept
{
	std::lock_guard<std::mutex> lock(mutex_particlesMovement);
	if (particlesMovementMap.size() <= maxParticles)
		particlesMovementMap[particleId].emplace_back(position);
}
