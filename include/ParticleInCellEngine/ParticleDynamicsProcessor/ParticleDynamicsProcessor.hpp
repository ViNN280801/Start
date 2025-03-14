#ifndef PARTICLEDYNAMICSPROCESSOR_HPP
#define PARTICLEDYNAMICSPROCESSOR_HPP

#include <shared_mutex>

#include "FiniteElementMethod/Assemblers/GSMAssembler.hpp"
#include "Geometry/Mesh/Cubic/CubicGrid.hpp"
#include "Particle/Particle.hpp"
#include "ParticleInCellEngine/PICTypes.hpp"
#include "ParticleInCellEngine/ParticleDynamicsProcessor/ParticleMovementTracker.hpp"
#include "ParticleInCellEngine/ParticleDynamicsProcessor/ParticlePhysicsUpdater.hpp"
#include "ParticleInCellEngine/ParticleDynamicsProcessor/ParticleSettler.hpp"
#include "ParticleInCellEngine/ParticleDynamicsProcessor/ParticleSurfaceCollisionHandler.hpp"
#include "ParticleInCellEngine/ParticleDynamicsProcessor/StopModelingObserver.hpp"
#include "Utilities/ThreadedProcessor.hpp"

/**
 * @class ParticleDynamicsProcessor
 * @brief Handles the processing of particle dynamics in a plasma environment.
 *
 * This class performs computations for:
 * - **Electromagnetic forces** acting on particles
 * - **Particle movement tracking**
 * - **Gas collisions** using various scattering models (HS, VHS, VSS)
 * - **Surface interactions** (settling, collisions)
 * - **Multi-threaded execution** using `std::execution::par` and OpenMP (if enabled)
 * - **CUDA-accelerated execution** (if CUDA is enabled)
 *
 * @note The implementation chooses the best available acceleration method:
 *       1. CUDA (if available and enabled via environment variable START_USE_CUDA=1)
 *       2. OpenMP (if available and numThreads > 1)
 *       3. Standard C++ parallel execution (fallback)
 */
class ParticleDynamicsProcessor
{
private:
    /**
     * @brief Helper function for parallel particle processing using C++ Standard Parallel Algorithms.
     *
     * @param start_index Starting index of the particle batch.
     * @param end_index Ending index of the particle batch.
     * @param timeMoment Current simulation time.
     * @param timeStep Time step for integration.
     * @param particles Vector of particles.
     * @param cubicGrid Pointer to the simulation grid.
     * @param gsmAssembler Pointer to the finite element assembler.
     * @param surfaceMesh Reference to the surface mesh for collision detection.
     * @param particleTracker Map for tracking particle movements.
     * @param settledParticlesIds Set of settled particle IDs.
     * @param sh_mutex_settledParticlesCounterMap Shared mutex for particle counter.
     * @param mutex_particlesMovementMap Mutex for movement tracking.
     * @param particleMovementMap Map storing particle movement history.
     * @param stopSubject Reference to the stop condition handler.
     * @param scatteringModel The scattering model used for gas collisions.
     * @param gasName The type of gas used for collisions.
     * @param gasConcentration The concentration of gas molecules.
     */
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

    /**
     * @brief Processes particle dynamics in parallel using standard algorithms.
     *
     * @param numThreads Number of threads to use.
     * @param timeMoment Current simulation time.
     * @param timeStep Time step for integration.
     * @param particles Vector of particles.
     */
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
    /**
     * @brief Processes particle dynamics in parallel using OpenMP.
     *
     * @param numThreads Number of threads to use.
     * @param timeMoment Current simulation time.
     * @param timeStep Time step for integration.
     * @param particles Vector of particles.
     */
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
    /**
     * @brief Main function to process particles using either GPU (CUDA), OpenMP, or standard parallel execution.
     *
     * The function selects the best available processing method in the following order of preference:
     * 1. OpenMP (if USE_OMP is defined and numThreads > 1)
     * 2. Standard C++ parallel algorithms (fallback)
     *
     * @param config_filename Name of the configuration file.
     * @param particles Vector of particles.
     * @param timeMoment Current simulation time.
     * @param timeStep Time step for integration.
     * @param numThreads Number of threads to use.
     * @param cubicGrid Pointer to the simulation grid.
     * @param gsmAssembler Pointer to the finite element assembler.
     * @param surfaceMesh Reference to the surface mesh for collision detection.
     * @param particleTracker Map for tracking particle movements.
     * @param settledParticlesIds Set of settled particle IDs.
     * @param sh_mutex_settledParticlesCounterMap Shared mutex for particle counter.
     * @param mutex_particlesMovementMap Mutex for movement tracking.
     * @param particleMovementMap Map storing particle movement history.
     * @param stopSubject Reference to the stop condition handler.
     * @param scatteringModel The scattering model used for gas collisions.
     * @param gasName The type of gas used for collisions.
     * @param gasConcentration The concentration of gas molecules.
     */
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
