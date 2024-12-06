#ifndef DYNAMICSOLVER_HPP
#define DYNAMICSOLVER_HPP

#include "Geometry/CubicGrid.hpp"
#include "ParticleInCellEngine/DynamicSolver/ParticleMovementTracker.hpp"
#include "ParticleInCellEngine/DynamicSolver/ParticlePhysicsUpdater.hpp"
#include "ParticleInCellEngine/DynamicSolver/ParticleSettler.hpp"
#include "ParticleInCellEngine/DynamicSolver/SurfaceCollisionHandler.hpp"
#include "Utilities/ThreadedProcessor.hpp"

/**
 * @class DynamicSolver
 * @brief Encapsulates the dynamics simulation loop for particle tracking and updates.
 *
 * This class provides methods to process particles in a simulation, including:
 * - Electromagnetic force updates
 * - Gas collision handling
 * - Surface collision detection
 * - Particle movement tracking
 * - Particle settlement updates
 *
 * It utilizes parallel execution to handle particle operations efficiently.
 */
class DynamicSolver
{
private:
    ParticleSettler m_particleSettler;          ///< Handles particle settlement on surfaces.
    ParticlePhysicsUpdater m_physicsUpdater;    ///< Updates particle physical properties.
    ParticleMovementTracker m_movementTracker;  ///< Tracks particle movements.
    SurfaceCollisionHandler m_collisionHandler; ///< Detects and handles surface collisions.

    /**
     * @brief Helper function for processing a segment of particles using standard parallel execution.
     * @details This function processes a subset of the particle vector (from start_index to end_index)
     *          using a parallel execution policy. It applies electromagnetic forces, handles gas and
     *          surface collisions, and updates particle movements. Any errors are caught and logged.
     *
     * @param start_index The starting index of the particle subset to process.
     * @param end_index The ending index (exclusive) of the particle subset to process.
     * @param timeMoment The current simulation time.
     * @param particles A reference to the vector of particles to process.
     * @param cubicGrid A shared pointer to the cubic grid used for spatial indexing.
     * @param gsmAssembler A shared pointer to the GSM assembler for mesh data access.
     * @param particleTracker A reference to the particle tracker map.
     */
    void _process_stdver__helper(size_t start_index,
                                 size_t end_index,
                                 double timeMoment,
                                 ParticleVector &particles,
                                 std::shared_ptr<CubicGrid> cubicGrid,
                                 std::shared_ptr<GSMAssembler> gsmAssembler,
                                 ParticleTrackerMap &particleTracker);

    /**
     * @brief Processes all particles using standard parallel execution.
     * @details This method divides the work among multiple threads using the ThreadedProcessor utility,
     *          which further splits the particles into segments and calls `_process_stdver__helper`
     *          for each segment. The parallel execution is deferred until the result is needed.
     *
     * @param numThreads The number of threads to use for parallel processing.
     * @param timeMoment The current simulation time.
     * @param particles The vector of particles to process.
     * @param cubicGrid A shared pointer to the cubic grid used for spatial indexing.
     * @param gsmAssembler A shared pointer to the GSM assembler for mesh data access.
     * @param particleTracker A reference to the particle tracker map.
     */
    void _process_stdver__(unsigned int numThreads,
                           double timeMoment,
                           ParticleVector &particles,
                           std::shared_ptr<CubicGrid> cubicGrid,
                           std::shared_ptr<GSMAssembler> gsmAssembler,
                           ParticleTrackerMap &particleTracker);

#ifdef USE_OMP
    /**
     * @brief Processes all particles using OpenMP parallelization.
     * @details This function uses OpenMP directives to distribute particle processing across multiple threads.
     *          It performs the same operations as `_process_stdver__` but without the ThreadedProcessor.
     *          Instead, it relies on OpenMP's `#pragma omp parallel for` to loop over the particles.
     *
     *          Each thread applies electromagnetic forces to particles, simulates gas collisions, checks
     *          for surface collisions, and records their movements. By leveraging OpenMP, this method
     *          can achieve improved performance on machines with multiple cores.
     *
     * @param numThreads The number of OpenMP threads to use for parallel processing.
     * @param timeMoment The current simulation time.
     * @param particles The vector of particles to process.
     * @param cubicGrid A shared pointer to the cubic grid used for spatial indexing.
     * @param gsmAssembler A shared pointer to the GSM assembler for mesh data access.
     * @param particleTracker A reference to the particle tracker map.
     */
    void _process_ompver__(unsigned int numThreads,
                           double timeMoment,
                           ParticleVector &particles,
                           std::shared_ptr<CubicGrid> cubicGrid,
                           std::shared_ptr<GSMAssembler> gsmAssembler,
                           ParticleTrackerMap &particleTracker);
#endif // !USE_OMP

public:
    /**
     * @brief Constructs the DynamicSolver, initializing its dependencies.
     * @param config_filename The configuration file for the simulation.
     * @param settledParticlesMutex Mutex for accessing settled particle data.
     * @param particlesMovementMutex Mutex for accessing particle movement data.
     * @param surfaceMeshAABBtree Reference to the AABB tree for collision detection.
     * @param triangleMesh Vector containing mesh triangle parameters.
     * @param settledParticlesIds Set of IDs of settled particles.
     * @param settledParticlesCounterMap Counter map for settled particles on triangles.
     * @param particlesMovement Map tracking particle movements.
     */
    DynamicSolver(std::string_view config_filename,
                  std::shared_mutex &settledParticlesMutex,
                  std::mutex &particlesMovementMutex,
                  AABB_Tree_Triangle const &surfaceMeshAABBtree,
                  MeshTriangleParamVector const &triangleMesh,
                  ParticlesIDSet &settledParticlesIds,
                  SettledParticlesCounterMap &settledParticlesCounterMap,
                  ParticleMovementMap &particlesMovement);

    /**
     * @brief Processes all particles at the given time moment.
     * @details Based on the compiled configuration, this method uses either the standard version or
     *          the OpenMP-accelerated version to process particles. It applies electromagnetic pushes,
     *          handles gas and surface collisions, and updates particle movements accordingly.
     *
     * @param config_filename The configuration file used for simulation parameters.
     * @param particles The vector of particles to process.
     * @param timeMoment The current simulation time.
     * @param cubicGrid A shared pointer to the cubic grid used for spatial indexing.
     * @param gsmAssembler A shared pointer to the GSM assembler for mesh data access.
     * @param particleTracker A reference to the particle tracker map.
     */
    void process(std::string_view config_filename,
                 ParticleVector &particles,
                 double timeMoment,
                 std::shared_ptr<CubicGrid> cubicGrid,
                 std::shared_ptr<GSMAssembler> gsmAssembler,
                 ParticleTrackerMap &particleTracker);
};

#endif // !DYNAMICSOLVER_HPP
