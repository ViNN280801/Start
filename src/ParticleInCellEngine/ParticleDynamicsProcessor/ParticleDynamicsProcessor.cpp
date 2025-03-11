#include <execution>

#ifdef USE_OMP
#include <omp.h>
#endif

#include "ParticleInCellEngine/ParticleDynamicsProcessor/ParticleDynamicsProcessor.hpp"

#ifdef USE_CUDA
#include "ParticleInCellEngine/ParticleDynamicsProcessor/CUDA/ParticleDynamicsProcessorCUDA.cuh"
#endif

void ParticleDynamicsProcessor::_process_stdver__helper(size_t start_index,
                                                        size_t end_index,
                                                        double timeMoment,
                                                        double timeStep,
                                                        ParticleVector &particles,
                                                        std::shared_ptr<CubicGrid> cubicGrid,
                                                        std::shared_ptr<GSMAssembler> gsmAssembler,
                                                        SurfaceMesh &surfaceMesh,
                                                        ParticleTrackerMap &particleTracker,
                                                        ParticlesIDSet &settledParticlesIds,
                                                        std::shared_mutex &sh_mutex_settledParticlesCounterMap,
                                                        std::mutex &mutex_particlesMovementMap,
                                                        ParticleMovementMap &particleMovementMap,
                                                        StopSubject &stopSubject,
                                                        std::string_view scatteringModel,
                                                        std::string_view gasName,
                                                        double gasConcentration)
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
                      [&cubicGrid, gsmAssembler, timeMoment, timeStep, &particleTracker, &particles,
                       &settledParticlesIds, &sh_mutex_settledParticlesCounterMap, &particleMovementMap,
                       &mutex_particlesMovementMap, scatteringModel, gasName, gasConcentration,
                       &surfaceMesh, &stopSubject](Particle &particle)
                      {
                          // 1. If particle is already settled, skip further processing.
                          if (ParticleSettler::isSettled(particle.getId(), settledParticlesIds, sh_mutex_settledParticlesCounterMap))
                              return;

                          // 2. Apply electromagnetic forces to the particle.
                          if (auto tetraId{CubicGrid::getContainingTetrahedron(particleTracker, particle, timeMoment)})
                              ParticlePhysicsUpdater::doElectroMagneticPush(particle, gsmAssembler, *tetraId, timeStep);

                          // 3. Record the particle's previous position for movement tracking.
                          Point const &prev{particle.getCentre()};
                          ParticleMovementTracker::recordMovement(particleMovementMap, mutex_particlesMovementMap, particle.getId(), prev);

                          // 4. Update the particle's position.
                          particle.updatePosition(timeStep);
                          Ray ray(prev, particle.getCentre());
                          if (ray.is_degenerate())
                              return;

                          // 5. Simulate collisions between the particle and gas molecules.
                          ParticlePhysicsUpdater::collideWithGas(particle, scatteringModel, gasName, gasConcentration, timeStep);

                          // 6. Handle collisions between the particle and the surface mesh.
                          auto collision{ParticleSurfaceCollisionHandler::handle(
                              particle, ray, particles.size(), surfaceMesh, sh_mutex_settledParticlesCounterMap,
                              mutex_particlesMovementMap, particleMovementMap, settledParticlesIds, stopSubject)};

                          // 7. If we collided and settled with a tetrahedron or triangle, we need to update collision counter.
                          if (collision)
                          {
                              if (stopSubject.isStopRequested())
                                  return;
                          }
                      });
    }
    catch (std::exception const &e)
    {
        throw std::runtime_error{std::string{"[process_stdver__helper] "} + e.what()};
    }
}

void ParticleDynamicsProcessor::_process_stdver__(unsigned int numThreads,
                                                  double timeMoment,
                                                  double timeStep,
                                                  ParticleVector &particles,
                                                  std::shared_ptr<CubicGrid> cubicGrid,
                                                  std::shared_ptr<GSMAssembler> gsmAssembler,
                                                  SurfaceMesh &surfaceMesh,
                                                  ParticleTrackerMap &particleTracker,
                                                  ParticlesIDSet &settledParticlesIds,
                                                  std::shared_mutex &sh_mutex_settledParticlesCounterMap,
                                                  std::mutex &mutex_particlesMovementMap,
                                                  ParticleMovementMap &particleMovementMap,
                                                  StopSubject &stopSubject,
                                                  std::string_view scatteringModel,
                                                  std::string_view gasName,
                                                  double gasConcentration)
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
        [](size_t start_index,
           size_t end_index,
           double timeMoment,
           double timeStep,
           ParticleVector &particles,
           std::shared_ptr<CubicGrid> cubicGrid,
           std::shared_ptr<GSMAssembler> gsmAssembler,
           SurfaceMesh &surfaceMesh,
           ParticleTrackerMap &particleTracker,
           ParticlesIDSet &settledParticlesIds,
           std::shared_mutex &sh_mutex_settledParticlesCounterMap,
           std::mutex &mutex_particlesMovementMap,
           ParticleMovementMap &particleMovementMap,
           StopSubject &stopSubject,
           std::string_view scatteringModel,
           std::string_view gasName,
           double gasConcentration)
        {
            _process_stdver__helper(start_index, end_index, timeMoment, timeStep,
                                    particles, cubicGrid, gsmAssembler, surfaceMesh,
                                    particleTracker, settledParticlesIds,
                                    sh_mutex_settledParticlesCounterMap, mutex_particlesMovementMap,
                                    particleMovementMap, stopSubject, scatteringModel, gasName, gasConcentration);
        },
        timeMoment, timeStep, std::ref(particles), cubicGrid, gsmAssembler, std::ref(surfaceMesh),
        std::ref(particleTracker), std::ref(settledParticlesIds), std::ref(sh_mutex_settledParticlesCounterMap),
        std::ref(mutex_particlesMovementMap), std::ref(particleMovementMap), std::ref(stopSubject),
        scatteringModel, gasName, gasConcentration);
}

#ifdef USE_OMP
void ParticleDynamicsProcessor::_process_ompver__(double timeMoment,
                                                  double timeStep,
                                                  unsigned int numThreads,
                                                  ParticleVector &particles,
                                                  std::shared_ptr<CubicGrid> cubicGrid,
                                                  std::shared_ptr<GSMAssembler> gsmAssembler,
                                                  SurfaceMesh &surfaceMesh,
                                                  ParticleTrackerMap &particleTracker,
                                                  ParticlesIDSet &settledParticlesIds,
                                                  std::shared_mutex &sh_mutex_settledParticlesCounterMap,
                                                  std::mutex &mutex_particlesMovementMap,
                                                  ParticleMovementMap &particleMovementMap,
                                                  StopSubject &stopSubject,
                                                  std::string_view scatteringModel,
                                                  std::string_view gasName,
                                                  double gasConcentration)
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
            if (ParticleSettler::isSettled(particle.getId(), settledParticlesIds, sh_mutex_settledParticlesCounterMap))
                continue;

            // 2. Electromagnetic push.
            if (auto tetraId{CubicGrid::getContainingTetrahedron(particleTracker, particle, timeMoment)})
                ParticlePhysicsUpdater::doElectroMagneticPush(particle, gsmAssembler, *tetraId, timeStep);

            // 3. Record previous position.
            Point prev{particle.getCentre()};
            ParticleMovementTracker::recordMovement(particleMovementMap, mutex_particlesMovementMap, particle.getId(), prev);

            // 4. Update position.
            particle.updatePosition(timeStep);
            Ray ray(prev, particle.getCentre());
            if (ray.is_degenerate())
                continue;

            // 5. Gas collision.
            ParticlePhysicsUpdater::collideWithGas(particle, scatteringModel, gasName, gasConcentration, timeStep);

            // 6. Skip surface collision checks if t == 0.
            if (timeMoment == 0.0)
                continue;

            // 7. Surface collision.
            ParticleSurfaceCollisionHandler::handle(particle, ray, particles.size(),
                                                    surfaceMesh, sh_mutex_settledParticlesCounterMap,
                                                    mutex_particlesMovementMap,
                                                    particleMovementMap, settledParticlesIds, stopSubject);
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
                                        double timeStep,
                                        unsigned int numThreads,
                                        std::shared_ptr<CubicGrid> cubicGrid,
                                        std::shared_ptr<GSMAssembler> gsmAssembler,
                                        SurfaceMesh &surfaceMesh,
                                        ParticleTrackerMap &particleTracker,
                                        ParticlesIDSet &settledParticlesIds,
                                        std::shared_mutex &sh_mutex_settledParticlesCounterMap,
                                        std::mutex &mutex_particlesMovementMap,
                                        ParticleMovementMap &particleMovementMap,
                                        StopSubject &stopSubject,
                                        std::string_view scatteringModel,
                                        std::string_view gasName,
                                        double gasConcentration)
{
#ifdef USE_CUDA
    try
    {
        LOGMSG("Using CUDA");
        int result = system("nvidia-smi");
        if (result != 0)
        {
            ERRMSG("Failed to execute nvidia-smi");
        }
        ParticleDynamicsProcessorCUDA::process(
            timeMoment, timeStep, particles, cubicGrid, gsmAssembler,
            surfaceMesh, particleTracker, settledParticlesIds,
            sh_mutex_settledParticlesCounterMap, mutex_particlesMovementMap,
            particleMovementMap, stopSubject, scatteringModel, gasName, gasConcentration);
        return;
    }
    catch (const std::exception &e)
    {
        // If CUDA processing fails, fall back to CPU processing
        ERRMSG(util::stringify("CUDA processing failed: ", e.what()));
    }
#endif

#ifdef USE_OMP
    // Check if we should use OpenMP first
    if (numThreads > 1)
    {
        LOGMSG(util::stringify("Using OpenMP with ", numThreads, " threads"));
        _process_ompver__(timeMoment, timeStep, numThreads, particles, cubicGrid, gsmAssembler,
                          surfaceMesh, particleTracker, settledParticlesIds, sh_mutex_settledParticlesCounterMap,
                          mutex_particlesMovementMap, particleMovementMap, stopSubject, scatteringModel,
                          gasName, gasConcentration);
        return;
    }
#endif

    // Fall back to standard C++ parallel execution
    LOGMSG(util::stringify("Using standard C++ parallel execution with ", numThreads, " threads"));
    _process_stdver__(numThreads, timeMoment, timeStep, particles, cubicGrid, gsmAssembler,
                      surfaceMesh, particleTracker, settledParticlesIds, sh_mutex_settledParticlesCounterMap,
                      mutex_particlesMovementMap, particleMovementMap, stopSubject, scatteringModel,
                      gasName, gasConcentration);
}
