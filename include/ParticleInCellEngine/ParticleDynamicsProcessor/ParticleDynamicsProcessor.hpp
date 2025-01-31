#ifndef PARTICLEDYNAMICSPROCESSOR_HPP
#define PARTICLEDYNAMICSPROCESSOR_HPP

#include "FiniteElementMethod/GSMAssembler.hpp"
#include "Geometry/CubicGrid.hpp"
#include "Particle/Particle.hpp"
#include "ParticleInCellEngine/PICTypes.hpp"

class ParticleDynamicsProcessor
{
private:
    static void _process_stdver__helper(size_t start_index,
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
                                        double gasConcentration);

    static void _process_stdver__(unsigned int numThreads,
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
                                  double gasConcentration);

#ifdef USE_OMP

    static void _process_ompver__(double timeMoment,
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
                           double gasConcentration);
#endif // !USE_OMP

public:
    static void process(std::string_view config_filename,
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
                 double gasConcentration);
};

#endif // !PARTICLEDYNAMICSPROCESSOR_HPP
