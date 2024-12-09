#include "ParticleInCellEngine/ParticleDynamicsProcessor/ParticleSettler.hpp"

ParticleSettler::ParticleSettler(std::shared_mutex &mutex,
                                 ParticlesIDSet &ids,
                                 SettledParticlesCounterMap &counterMap)
    : m_settledParticlesMutex(mutex),
      m_settledParticleIds(ids),
      m_settledParticlesCounterMap(counterMap) {}

bool ParticleSettler::isSettled(size_t particleId)
{
    std::shared_lock<std::shared_mutex> lock(m_settledParticlesMutex);
    return m_settledParticleIds.find(particleId) != m_settledParticleIds.end();
}

void ParticleSettler::settle(size_t particleId, size_t triangleId, size_t totalParticles)
{
    std::unique_lock<std::shared_mutex> lock(m_settledParticlesMutex);
    ++m_settledParticlesCounterMap[triangleId];
    m_settledParticleIds.insert(particleId);

    if (m_settledParticleIds.size() >= totalParticles)
        throw std::runtime_error("All particles are settled");
}
