#include <algorithm>
#include <atomic>
#include <execution>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "DataHandling/TriangleMeshHdf5Manager.hpp"
#include "FiniteElementMethod/BoundaryConditions/BoundaryConditionsManager.hpp"
#include "FiniteElementMethod/FEMCheckers.hpp"
#include "FiniteElementMethod/FEMInitializer.hpp"
#include "FiniteElementMethod/FEMLimits.hpp"
#include "FiniteElementMethod/FEMPrinter.hpp"
#include "Generators/ParticleGenerator.hpp"
#include "ModelingMainDriver.hpp"
#include "ParticleInCellEngine/ChargeDensityEquationSolver.hpp"
#include "ParticleInCellEngine/NodeChargeDensityProcessor.hpp"
#include "ParticleInCellEngine/ParticleDynamicsProcessor/ParticleDynamicsProcessor.hpp"

#ifdef USE_CUDA
#include "Particle/CUDA/ParticleDeviceMemoryConverter.cuh"
#endif

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
    if (m_config.isParticleSourcePoint())
    {
        auto tmp{ParticleGenerator::fromPointSource(m_config.getParticleSourcePoints())};
#ifdef USE_CUDA
        ParticleVector h_particles{ParticleDeviceMemoryConverter::copyToHost(tmp)};
        if (!h_particles.empty())
            m_particles.insert(m_particles.end(), std::begin(h_particles), std::end(h_particles));
#endif
        m_particles.insert(m_particles.end(), std::begin(tmp), std::end(tmp));
    }
    if (m_config.isParticleSourceSurface())
    {
        auto tmp{ParticleGenerator::fromSurfaceSource(m_config.getParticleSourceSurfaces())};
#ifdef USE_CUDA
        ParticleVector h_particles{ParticleDeviceMemoryConverter::copyToHost(tmp)};
        if (!h_particles.empty())
            m_particles.insert(m_particles.end(), std::begin(h_particles), std::end(h_particles));
#endif
        m_particles.insert(m_particles.end(), std::begin(tmp), std::end(tmp));
    }

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
      m_surfaceMesh(m_config.getMeshFilename(), "Substrate")
{
    // Checking mesh filename on validity and assign it to the class member.
    FEMCheckers::checkMeshFile(m_config.getMeshFilename());

    // Calculating and checking gas concentration.
    m_gasConcentration = util::calculateConcentration_w(config_filename);

    // Global initializator. Initializes stop obervers and spawning particles.
    _ginitialize();
}

ModelingMainDriver::~ModelingMainDriver() { _gfinalize(); }

void ModelingMainDriver::startModeling()
{
    double timeStep{m_config.getTimeStep()};
    double totalTime{m_config.getSimulationTime()};
    unsigned int numThreads{m_config.getNumThreads_s()};

    for (double timeMoment{}; timeMoment <= totalTime && !m_stop_processing.test(); timeMoment += timeStep)
    {
        try
        {
            omp_set_num_threads(numThreads);

#pragma omp parallel for schedule(dynamic)
            for (size_t i = 0ul; i < m_particles.size(); ++i)
            {
                auto &particle = m_particles[i];

                // 1. Check if particle is settled.
                if (ParticleSettler::isSettled(particle.getId(), m_settledParticlesIds, m_settledParticlesMutex))
                    continue;

                // 2. Electromagnetic push (Here is no need to do EM-push).
                // @remark In the sputtering model there is no any field, so, there is no need to do EM-push.
                // m_physicsUpdater.doElectroMagneticPush(particle, gsmAssembler, *tetrahedronId);

                // 3. Record previous position.
                Point prev{particle.getCentre()};
                ParticleMovementTracker::recordMovement(m_particlesMovement, m_particlesMovementMutex, particle.getId(), prev);

                // 4. Update position.
                particle.updatePosition(timeStep);
                Ray ray(prev, particle.getCentre());

                // 5. Gas collision.
                ParticlePhysicsUpdater::collideWithGas(particle, m_config.getScatteringModel(), m_config.getGasStr(), m_gasConcentration, timeStep);

                // 6. Skip surface collision checks if t == 0.
                if (timeMoment == 0.0)
                    continue;

                // 7. Surface collision.
                ParticleSurfaceCollisionHandler::handle(particle,
                                                        ray,
                                                        m_particles.size(),
                                                        m_surfaceMesh,
                                                        m_settledParticlesMutex,
                                                        m_particlesMovementMutex,
                                                        m_particlesMovement,
                                                        m_settledParticlesIds,
                                                        *this);
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

        if (m_stop_processing.test())
        {
            SUCCESSMSG(util::stringify("All particles are settled. Stop requested by observers, terminating the simulation loop. ",
                                       "Last time moment is: ", timeMoment, "s."));
        }
        SUCCESSMSG(util::stringify("Time = ", timeMoment, "s. Totally settled: ", m_settledParticlesIds.size(), "/", m_particles.size(), " particles."));
    }
}
