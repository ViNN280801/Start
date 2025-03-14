#include <algorithm>
#include <atomic>
#include <execution>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "DataHandling/TriangleMeshHdf5Manager.hpp"
#include "FiniteElementMethod/BoundaryConditions/BoundaryConditionsManager.hpp"
#include "FiniteElementMethod/Utils/FEMCheckers.hpp"
#include "FiniteElementMethod/Utils/FEMInitializer.hpp"
#include "FiniteElementMethod/Utils/FEMLimits.hpp"
#include "FiniteElementMethod/Utils/FEMPrinter.hpp"
#include "Generators/ParticleGenerator.hpp"
#include "ModelingMainDriver.hpp"
#include "ParticleInCellEngine/ChargeDensityEquationSolver.hpp"
#include "ParticleInCellEngine/NodeChargeDensityProcessor.hpp"
#include "ParticleInCellEngine/ParticleDynamicsProcessor/ParticleDynamicsProcessor.hpp"
#include "Utilities/GmshUtilities/GmshUtils.hpp"

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

void ModelingMainDriver::_initializeParticles()
{
    ParticleVector pointSourceParticles;
    ParticleVector surfaceSourceParticles;

    // Get particles directly as host particles
    if (m_config.isParticleSourcePoint())
    {
#ifdef USE_CUDA
        pointSourceParticles = ParticleGeneratorDevice::fromPointSource(m_config.getParticleSourcePoints());
#else
        pointSourceParticles = ParticleGenerator::fromPointSource(m_config.getParticleSourcePoints());
#endif
    }

    if (m_config.isParticleSourceSurface())
    {
#ifdef USE_CUDA
        surfaceSourceParticles = ParticleGeneratorDevice::fromSurfaceSource(m_config.getParticleSourceSurfaces());
#else
        surfaceSourceParticles = ParticleGenerator::fromSurfaceSource(m_config.getParticleSourceSurfaces());
#endif
    }

    m_particles.reserve(m_particles.size() + pointSourceParticles.size() + surfaceSourceParticles.size());
    m_particles.insert(m_particles.end(), pointSourceParticles.begin(), pointSourceParticles.end());
    m_particles.insert(m_particles.end(), surfaceSourceParticles.begin(), surfaceSourceParticles.end());

    if (m_particles.empty())
        throw std::runtime_error("Particles are uninitialized, check your configuration file");
}

void ModelingMainDriver::_ginitialize()
{
    _initializeObservers();
    _initializeParticles();
}

void ModelingMainDriver::_updateSurfaceMesh()
{
    // Updating hdf5file to know how many particles settled on certain triangle from the surface mesh.
    std::string hdf5filename(std::string(m_config.getMeshFilename().substr(0ul, m_config.getMeshFilename().find("."))));
    hdf5filename += ".hdf5";
    TriangleMeshHdf5Manager hdf5handler(hdf5filename);
    hdf5handler.saveMeshToHDF5(m_surfaceMesh.getTriangleCellMap());
}

void ModelingMainDriver::_saveParticleMovements() const
{
    try
    {
        if (m_particlesMovement.empty())
        {
            WARNINGMSG("Warning: Particle movements map is empty, no data to save");
            return;
        }

        json j;
        for (auto const &[id, movements] : m_particlesMovement)
        {
            if (movements.size() > 1)
            {
                json positions;
                for (auto const &point : movements)
                    positions.push_back({{"x", point.x()}, {"y", point.y()}, {"z", point.z()}});
                j[std::to_string(id)] = positions;
            }
            else
                throw std::runtime_error("There is no movements between particles, something may go wrong.");
        }

        std::string filepath("results/particles_movements.json");
        std::ofstream file(filepath);
        if (file.is_open())
        {
            file << j.dump(4); // 4 spaces indentation for pretty printing
            file.close();
        }
        else
            throw std::ios_base::failure("Failed to open file for writing");
        LOGMSG(util::stringify("Successfully written particle movements to the file ", filepath));

        util::check_json_validity(filepath);
    }
    catch (std::ios_base::failure const &e)
    {
        ERRMSG(util::stringify("I/O error occurred: ", e.what()));
    }
    catch (json::exception const &e)
    {
        ERRMSG(util::stringify("JSON error occurred: ", e.what()));
    }
    catch (std::runtime_error const &e)
    {
        ERRMSG(util::stringify("Error checking the just written file: ", e.what()));
    }
}

void ModelingMainDriver::_gfinalize()
{
    _updateSurfaceMesh();
    _saveParticleMovements();
}

ModelingMainDriver::ModelingMainDriver(std::string_view config_filename)
    : m_config_filename(config_filename),
      m_config(config_filename),
      m_surfaceMesh(m_config.getMeshFilename())
{
    // Checking mesh filename on validity and assign it to the class member.
    GmshUtils::checkGmshMeshFile(m_config.getMeshFilename());

    // Calculating and checking gas concentration.
    m_gasConcentration = util::calculateConcentration_w(config_filename);

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
    /* Ending of the FEM initialization. */

    /* == Beginning of the multithreading settings loading. == */
    double timeStep{m_config.getTimeStep()};
    double totalTime{m_config.getSimulationTime()};
    auto numThreads{m_config.getNumThreads_s()};
    size_t countOfParticles{m_particles.size()};
    /* ======================== End ========================== */

    std::map<GlobalOrdinal, double> nodeChargeDensityMap;
#if __cplusplus >= 202002L
    for (double timeMoment{}; timeMoment <= totalTime && !m_stop_processing.test(); timeMoment += timeStep)
#else
    for (double timeMoment{}; timeMoment <= totalTime && !m_stop_processing.test_and_set(); timeMoment += timeStep)
#endif
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

        // 3. Process all the particle dynamics in parallel (settling on surface,
        //    velocity and energy updater, EM-pusher and movement tracker).
        ParticleDynamicsProcessor::process(m_config_filename, m_particles, timeMoment, m_config.getTimeStep(),
                                           numThreads, cubicGrid, gsmAssembler, m_surfaceMesh, m_particleTracker, m_settledParticlesIds,
                                           m_settledParticlesMutex, m_particlesMovementMutex, m_particlesMovement,
                                           *this, m_config.getScatteringModel(), m_config.getGasStr(), m_gasConcentration);

#if __cplusplus >= 202002L
        if (m_stop_processing.test())
#else
        if (m_stop_processing.test_and_set())
#endif
        {
            SUCCESSMSG(util::stringify("All particles are settled. Stop requested by observers, terminating the simulation loop. ",
                                       "Last time moment is: ", timeMoment, "s."));
        }
        SUCCESSMSG(util::stringify("Time = ", timeMoment, "s. Totally settled: ", m_settledParticlesIds.size(), "/", countOfParticles, " particles."));
    }
}
