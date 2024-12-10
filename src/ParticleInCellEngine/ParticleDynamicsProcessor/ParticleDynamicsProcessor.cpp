#include <execution>

#ifdef USE_OMP
#include <omp.h>
#endif

#include "ParticleInCellEngine/ParticleDynamicsProcessor/ParticleDynamicsProcessor.hpp"
#include "Utilities/ConfigParser.hpp"

ParticleDynamicsProcessor::ParticleDynamicsProcessor(std::string_view config_filename,
                                                     std::shared_mutex &settledParticlesMutex,
                                                     std::mutex &particlesMovementMutex,
                                                     AABB_Tree_Triangle const &surfaceMeshAABBtree,
                                                     MeshTriangleParamVector const &triangleMesh,
                                                     ParticlesIDSet &settledParticlesIds,
                                                     SettledParticlesCounterMap &settledParticlesCounterMap,
                                                     ParticleMovementMap &particlesMovement)
    : m_particleSettler(settledParticlesMutex, settledParticlesIds, settledParticlesCounterMap),
      m_physicsUpdater(config_filename),
      m_movementTracker(particlesMovement, particlesMovementMutex),
      m_collisionHandler(settledParticlesMutex, particlesMovementMutex, surfaceMeshAABBtree,
                         triangleMesh, settledParticlesIds, settledParticlesCounterMap, particlesMovement) {}

void ParticleDynamicsProcessor::_process_stdver__helper(size_t start_index,
                                                        size_t end_index,
                                                        double timeMoment,
                                                        ParticleVector &particles,
                                                        std::shared_ptr<CubicGrid> cubicGrid,
                                                        std::shared_ptr<GSMAssembler> gsmAssembler,
                                                        ParticleTrackerMap &particleTracker)
{
    try
    {
#if __cplusplus >= 202002L
        std::for_each(std::execution::par_unseq,
#else
        std::for_each(std::execution::par,
#endif
                      particles.begin() + start_index,
                      particles.begin() + end_index,
                      [this, &cubicGrid, gsmAssembler, timeMoment, &particleTracker, &particles](auto &particle)
                      {
                          // 1. If particle is already settled, skip further processing.
                          if (m_particleSettler.isSettled(particle.getId()))
                              return;

                          // 2. Apply electromagnetic forces to the particle.
                          if (auto tetraId{CubicGrid::getContainingTetrahedron(particleTracker, particle, timeMoment)})
                              m_physicsUpdater.doElectroMagneticPush(particle, gsmAssembler, *tetraId);

                          // 3. Record the particle's previous position for movement tracking.
                          Point const prev{particle.getCentre()};
                          m_movementTracker.recordMovement(particle, prev);

                          // 4. Update the particle's position.
                          m_physicsUpdater.updatePosition(particle);
                          Ray ray(prev, particle.getCentre());
                          if (ray.is_degenerate())
                              return;

                          // 5. Simulate collisions between the particle and gas molecules.
                          m_physicsUpdater.collideWithGas(particle);

                          // 6. Skip surface collision checks at the initial simulation time (t == 0).
                          // 7. Detect and handle collisions with the surface mesh.
                          if (timeMoment != 0.0)
                              if (auto triangleId{m_collisionHandler.handle(particle, ray, particles.size())})
                                  m_particleSettler.settle(particle.getId(), *triangleId, particles.size());
                      });
    }
    catch (std::exception const &ex)
    {
        ERRMSG(util::stringify("Can't finish detecting particles collisions with surfaces: ", ex.what()));
    }
    catch (...)
    {
        ERRMSG("Some error occured while detecting particles collisions with surfaces");
    }
}

void ParticleDynamicsProcessor::_process_stdver__(unsigned int numThreads,
                                                  double timeMoment,
                                                  ParticleVector &particles,
                                                  std::shared_ptr<CubicGrid> cubicGrid,
                                                  std::shared_ptr<GSMAssembler> gsmAssembler,
                                                  ParticleTrackerMap &particleTracker)
{
    // We use `std::launch::deferred` here because surface collision tracking is not critical to be run immediately
    // after particle tracking. It can be deferred until it is needed, allowing for potential optimizations
    // such as saving resources when not required right away. Deferring this task ensures that the function
    // only runs when explicitly requested via `get()` or `wait()`, thereby reducing overhead if the results are
    // not immediately needed. This approach also helps avoid contention with more urgent tasks running in parallel.
    ThreadedProcessor::launch(
        particles.size(),
        numThreads,
        std::launch::deferred,
        [this](size_t start_index, size_t end_index, double timeMoment,
               ParticleVector &particles,
               std::shared_ptr<CubicGrid> cubicGrid,
               std::shared_ptr<GSMAssembler> gsmAssembler,
               ParticleTrackerMap &particleTracker)
        {
            this->_process_stdver__helper(start_index, end_index, timeMoment, particles, cubicGrid, gsmAssembler, particleTracker);
        },
        timeMoment, std::ref(particles), cubicGrid, gsmAssembler, std::ref(particleTracker));
}

#ifdef USE_OMP
void ParticleDynamicsProcessor::_process_ompver__(unsigned int numThreads,
                                                  double timeMoment,
                                                  ParticleVector &particles,
                                                  std::shared_ptr<CubicGrid> cubicGrid,
                                                  std::shared_ptr<GSMAssembler> gsmAssembler,
                                                  ParticleTrackerMap &particleTracker)
{
    try
    {
        // Set the number of OpenMP threads.
        omp_set_num_threads(numThreads);

#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0ul; i < particles.size(); ++i)
        {
            auto &particle = particles[i];

            // 1. Check if particle is settled.
            if (m_particleSettler.isSettled(particle.getId()))
                continue;

            // 2. Electromagnetic push.
            auto containingTetrahedronOpt = CubicGrid::getContainingTetrahedron(particleTracker, particle, timeMoment);
            if (containingTetrahedronOpt.has_value())
                m_physicsUpdater.doElectroMagneticPush(particle, gsmAssembler, containingTetrahedronOpt.value());

            // 3. Record previous position.
            Point prev{particle.getCentre()};
            m_movementTracker.recordMovement(particle, prev);

            // 4. Update position.
            m_physicsUpdater.updatePosition(particle);
            Ray ray(prev, particle.getCentre());
            if (ray.is_degenerate())
                continue;

            // 5. Gas collision.
            m_physicsUpdater.collideWithGas(particle);

            // 6. Skip surface collision checks if t == 0.
            if (timeMoment == 0.0)
                continue;

            // 7. Surface collision.
            auto triangleIdOpt{m_collisionHandler.handle(particle, ray, particles.size())};
            if (triangleIdOpt.has_value())
                m_particleSettler.settle(particle.getId(), triangleIdOpt.value(), particles.size());
        }
    }
    catch (std::exception const &ex)
    {
        ERRMSG(util::stringify("Can't finish detecting particles collisions with surfaces (OMP): ", ex.what()));
    }
    catch (...)
    {
        ERRMSG("Some error occured while detecting particles collisions with surfaces (OMP)");
    }
}
#endif // !USE_OMP

void ParticleDynamicsProcessor::process(std::string_view config_filename,
                                        ParticleVector &particles,
                                        double timeMoment,
                                        std::shared_ptr<CubicGrid> cubicGrid,
                                        std::shared_ptr<GSMAssembler> gsmAssembler,
                                        ParticleTrackerMap &particleTracker)
{
    // Check if the requested number of threads exceeds available hardware concurrency.
    ConfigParser configParser(config_filename);
    auto numThreads{configParser.getNumThreads_s()};

#ifdef USE_OMP
    // If OpenMP is available and requested, use the OMP version.
    _process_ompver__(numThreads, timeMoment, particles, cubicGrid, gsmAssembler, particleTracker);
#else
    // Otherwise, fall back to the standard version.
    _process_stdver__(numThreads, timeMoment, particles, cubicGrid, gsmAssembler, particleTracker);
#endif
}
