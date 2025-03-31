#include <algorithm>
#include <atomic>
#include <cmath>
#include <execution>
#include <limits>
#include <nlohmann/json.hpp>
#include <sstream>
using json = nlohmann::json;

#include "DataHandling/TriangleMeshHdf5Manager.hpp"
#include "FiniteElementMethod/BoundaryConditions/BoundaryConditionsManager.hpp"
#include "FiniteElementMethod/Utils/FEMCheckers.hpp"
#include "FiniteElementMethod/Utils/FEMInitializer.hpp"
#include "FiniteElementMethod/Utils/FEMLimits.hpp"
#include "FiniteElementMethod/Utils/FEMPrinter.hpp"
#include "Generators/ParticleGenerator.hpp"
#include "ModelingMainDriver.hpp"
#include "Particle/ParticleDisplacer.hpp"
#include "ParticleInCellEngine/ChargeDensityEquationSolver.hpp"
#include "ParticleInCellEngine/NodeChargeDensityProcessor.hpp"
#include "ParticleInCellEngine/ParticleDynamicsProcessor/ParticleDynamicsProcessor.hpp"
#include "Utilities/GmshUtilities/GmshUtils.hpp"

std::mutex ModelingMainDriver::m_particlesVectorMutex;
std::mutex ModelingMainDriver::m_PICTrackerMutex;
std::mutex ModelingMainDriver::m_nodeChargeDensityMapMutex;
std::mutex ModelingMainDriver::m_particlesMovementMutex;
std::shared_mutex ModelingMainDriver::m_settledParticlesMutex;
std::atomic_flag ModelingMainDriver::m_stop_processing = ATOMIC_FLAG_INIT;

void ModelingMainDriver::_initializeObservers()
{
    m_stopObserver = std::make_shared<StopFlagObserver>(m_stop_processing);
    addObserver(m_stopObserver);
}

void ModelingMainDriver::_spawnParticles()
{
    ParticleVector pointSourceParticles;
    ParticleVector surfaceSourceParticles;

    if (m_config.hasParticleSourcePoint())
    {
#ifdef USE_CUDA
        pointSourceParticles = ParticleGeneratorDevice::fromPointSource(m_config.getParticleSourcePoints());
#else
        pointSourceParticles = ParticleGenerator::fromPointSource(m_config.getParticleSourcePoints());
#endif
        ParticleDisplacer::displaceParticlesFromPointSources(pointSourceParticles, m_config.getParticleSourcePoints());
    }

    if (m_config.hasParticleSourceSurface())
    {
#ifdef USE_CUDA
        surfaceSourceParticles = ParticleGeneratorDevice::fromSurfaceSource(m_config.getParticleSourceSurfaces());
#else
        surfaceSourceParticles = ParticleGenerator::fromSurfaceSource(m_config.getParticleSourceSurfaces());
#endif
        ParticleDisplacer::displaceParticlesFromSurfaceSources(surfaceSourceParticles, m_config.getParticleSourceSurfaces());
    }

    // Now add the new particles to the main container
    // With stable_vector, this won't invalidate iterators to existing particles
    std::lock_guard<std::mutex> lock(m_particlesVectorMutex);
    m_particles.reserve(m_particles.size() + pointSourceParticles.size() + surfaceSourceParticles.size());
    m_particles.insert(m_particles.end(), pointSourceParticles.begin(), pointSourceParticles.end());
    m_particles.insert(m_particles.end(), surfaceSourceParticles.begin(), surfaceSourceParticles.end());

    if (m_particles.empty())
        throw std::runtime_error("Particles are uninitialized, check your configuration file");
}

void ModelingMainDriver::_ginitialize()
{
    _initializeObservers();
    _spawnParticles();
}

void ModelingMainDriver::_updateSurfaceMesh()
{
    // Updating hdf5file to know how many particles settled on certain triangle from the surface mesh.
    std::string hdf5filename(std::string(m_config.getMeshFilename().substr(0ul, m_config.getMeshFilename().find("."))));
    hdf5filename += ".hdf5";
    TriangleMeshHdf5Manager hdf5handler(hdf5filename);
    hdf5handler.saveMeshToHDF5(m_surfaceMesh.getTriangleCellMap());
}

void ModelingMainDriver::_gfinalize()
{
    _updateSurfaceMesh();
    ParticleMovementTracker::saveMovementsToJson(m_particlesMovement);
}

ModelingMainDriver::ModelingMainDriver(std::string_view config_filename)
    : m_config_filename(config_filename),
      m_config(config_filename),
      m_surfaceMesh(m_config.getMeshFilename())
{
    // Checking mesh filename on validity and assign it to the class member.
    GmshUtils::checkGmshMeshFile(m_config.getMeshFilename());

    // Calculating and checking gas concentration.
    m_gasConcentration = util::calculateConcentration(config_filename);

    // Global initializator. Initializes stop obervers and spawning particles.
    _ginitialize();
}

ModelingMainDriver::~ModelingMainDriver() { _gfinalize(); }

void ModelingMainDriver::startModeling()
{
    // =============================================
    // === Default main modeling  ==================
    // =============================================
    /* Beginning of the FEM initialization. */
    FEMInitializer feminit(m_config);
    auto gsmAssembler{feminit.getGlobalStiffnessMatrixAssembler()};
    auto cubicGrid{feminit.getCubicGrid()};
    auto boundaryConditions{feminit.getBoundaryConditions()};
    auto solutionVector{feminit.getEquationRHS()};
    std::map<GlobalOrdinal, double> nodeChargeDensityMap;
    /* Ending of the FEM initialization. */

    /* == Beginning of the multithreading settings loading. == */
    double timeStep{m_config.getTimeStep()};
    double totalTime{m_config.getSimulationTime()};
    auto numThreads{m_config.getNumThreads()};
    /* ======================== End ========================== */
#if __cplusplus >= 202002L
    for (double timeMoment{}; timeMoment <= totalTime && !m_stop_processing.test(); timeMoment += timeStep)
#else
    for (double timeMoment{}; timeMoment <= totalTime && !m_stop_processing.test_and_set(); timeMoment += timeStep)
#endif
    {
        // 1. Spawn new particles for this time step (continuous sputtering)
        // Time moment here not 0.0 because in _initialize() we already spawn particles
        // and we don't want to spawn particles in the last time step because
        // it will be handled in the next time step, so it doesn't make sense
        if (m_config.isSputtering() && timeMoment != 0.0 && timeMoment != totalTime)
            _spawnParticles();

        if (!m_config.isSputtering())
        {
            NodeChargeDensityProcessor::gather(timeMoment,
                                               m_config_filename,
                                               cubicGrid,
                                               gsmAssembler,
                                               m_particles,
                                               m_settledParticlesIds,
                                               m_particleTracker,
                                               nodeChargeDensityMap);

            // 2. Solve equation in the main thread.
            ChargeDensityEquationSolver::solve(timeMoment,
                                               m_config_filename,
                                               nodeChargeDensityMap,
                                               gsmAssembler,
                                               solutionVector,
                                               boundaryConditions);
        }

        // 3. Process all the particle dynamics in parallel (settling on surface,
        //    velocity and energy updater, EM-pusher and movement tracker).
        if (!m_config.isSputtering())
        {
            ParticleDynamicsProcessor::process(m_config_filename,
                                               m_particles,
                                               timeMoment,
                                               m_config.getTimeStep(),
                                               numThreads,
                                               cubicGrid,
                                               gsmAssembler,
                                               m_surfaceMesh,
                                               m_particleTracker,
                                               m_settledParticlesIds,
                                               m_settledParticlesMutex,
                                               m_particlesMovementMutex,
                                               m_particlesMovement,
                                               *this,
                                               m_config.getScatteringModel(),
                                               m_config.getGasStr(),
                                               m_gasConcentration);
        }
        else
        {
            ParticleDynamicsProcessor::process(m_config_filename,
                                               m_particles,
                                               timeMoment,
                                               m_config.getTimeStep(),
                                               numThreads,
                                               nullptr, // cubicGrid
                                               nullptr, // gsmAssembler
                                               m_surfaceMesh,
                                               m_particleTracker,
                                               m_settledParticlesIds,
                                               m_settledParticlesMutex,
                                               m_particlesMovementMutex,
                                               m_particlesMovement,
                                               *this,
                                               m_config.getScatteringModel(),
                                               m_config.getGasStr(),
                                               m_gasConcentration);
        }

#if __cplusplus >= 202002L
        if (m_stop_processing.test())
#else
        if (m_stop_processing.test_and_set())
#endif
        {
            SUCCESSMSG(util::stringify("All particles are settled. Stop requested by observers, terminating the simulation loop. ",
                                       "Last time moment is: ", timeMoment, "s."));
        }
        SUCCESSMSG(util::stringify("Time = ", timeMoment,
                                   "s. Totally settled: ",
                                   m_settledParticlesIds.size(), "/",
                                   m_particles.size(), " particles."));
    }
}
