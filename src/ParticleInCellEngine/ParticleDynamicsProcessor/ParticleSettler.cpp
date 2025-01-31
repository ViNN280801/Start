#include "ParticleInCellEngine/ParticleDynamicsProcessor/ParticleSettler.hpp"

bool ParticleSettler::isSettled(size_t particleId,
                                ParticlesIDSet const &settledParticlesIds,
                                std::shared_mutex &sh_mutex_settledParticlesCounterMap) { return settledParticlesIds.find(particleId) != settledParticlesIds.cend(); }

void ParticleSettler::settle(size_t particleId,
                             size_t triangleId,
                             size_t totalParticles,
                             SurfaceMesh &surfaceMesh,
                             std::shared_mutex &sh_mutex_settledParticlesCounterMap,
                             ParticlesIDSet &settledParticlesIds,
                             StopSubject &stopSubject)
{
    std::unique_lock<std::shared_mutex> lock(sh_mutex_settledParticlesCounterMap);

    // 1. Increasing counter of settled particles on triangle with ID == 'triangleId'.
    surfaceMesh.getTriangleCellMap()[triangleId].count += 1;

    // 2. Adding ID of the settled particle to have an opportunity not to process it in further processes.
    settledParticlesIds.insert(particleId);

    // 3. If current settled particles count >= count of particles - we need to stop the modeling process.
    if (surfaceMesh.getTotalCountOfSettledParticles() >= totalParticles)
    {
        // Notifying stop object (atomic_flag) to stop main modeling loop.
        stopSubject.notifyStopRequested();
        return;
    }
}
