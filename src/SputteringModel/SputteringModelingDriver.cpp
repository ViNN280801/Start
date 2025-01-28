#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "DataHandling/TriangleMeshHdf5Manager.hpp"
#include "ParticleInCellEngine/ParticleDynamicsProcessor/ParticleDynamicsProcessor.hpp"
#include "SputteringModel/SputteringModelingDriver.hpp"

std::mutex SputteringModelingDriver::m_particlesMovementMutex;
std::shared_mutex SputteringModelingDriver::m_settledParticlesMutex;
std::atomic_flag SputteringModelingDriver::m_stop_processing = ATOMIC_FLAG_INIT;

void SputteringModelingDriver::_initializeObservers()
{
    m_stopObserver = std::make_shared<StopFlagObserver>(m_stop_processing);
    addObserver(m_stopObserver);
}

void SputteringModelingDriver::_ginitialize() { _initializeObservers(); }

void SputteringModelingDriver::_updateSurfaceMesh()
{
    // Updating hdf5file to know how many particles settled on certain triangle from the surface mesh.
    auto mapEnd{m_settledParticlesCounterMap.cend()};
    for (auto &[id, triangleCell] : m_surfaceMesh.getTriangleCellMap())
        if (auto it{m_settledParticlesCounterMap.find(id)}; it != mapEnd)
            triangleCell.count = it->second;

    std::string hdf5filename(std::string(m_mesh_filename.substr(0ul, m_mesh_filename.find("."))));
    hdf5filename += ".hdf5";
    TriangleMeshHdf5Manager hdf5handler(hdf5filename);
    hdf5handler.saveMeshToHDF5(m_surfaceMesh.getTriangleCellMap());
}

void SputteringModelingDriver::_saveParticleMovements() const
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

void SputteringModelingDriver::_gfinalize()
{
    _updateSurfaceMesh();
    _saveParticleMovements();
}

SputteringModelingDriver::SputteringModelingDriver(std::string_view mesh_filename) 
: m_mesh_filename(mesh_filename), m_surfaceMesh(Mesh::getMeshParams(mesh_filename)) { _ginitialize(); }

SputteringModelingDriver::~SputteringModelingDriver() { _gfinalize(); }

void SputteringModelingDriver::startModeling(double simtime, double timeStep, unsigned int numThreads,
                                             ParticleVector &particles, std::string_view scatteringModel,
                                             std::string_view gasName, double pressure, double temperature)
{
    double gasConcentration{(pressure / (constants::physical_constants::R * temperature)) * constants::physical_constants::N_av};
    if (gasConcentration < constants::gasConcentrationMinimalValue)
    {
        WARNINGMSG(util::stringify("Something wrong with the concentration of the gas. Its value is ", gasConcentration, ". Simulation might considerably slows down"));
    }

    for (double timeMoment{}; timeMoment <= simtime && !m_stop_processing.test(); timeMoment += timeStep)
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
                {
                    std::shared_lock<std::shared_mutex> lock(m_settledParticlesMutex);
                    if (m_settledParticlesIds.find(particle.getId()) != m_settledParticlesIds.end())
                        continue;
                }

                // 2. Electromagnetic push (Here is no need to do EM-push).
                // m_physicsUpdater.doElectroMagneticPush(particle, gsmAssembler, *tetrahedronId);

                // 3. Record previous position.
                Point prev{particle.getCentre()};
                {
                    std::lock_guard<std::mutex> lock(m_particlesMovementMutex);
                    if (m_particlesMovement.size() <= particles.size())
                        m_particlesMovement[particle.getId()].emplace_back(prev);
                }

                // 4. Update position.
                particle.updatePosition(timeStep);
                Ray ray(prev, particle.getCentre());
                if (ray.is_degenerate())
                    continue;

                // 5. Gas collision.
                auto collisionModel{CollisionModelFactory::create(scatteringModel)};
                collisionModel->collide(particle, util::getParticleTypeFromStrRepresentation(gasName), gasConcentration, timeStep);

                // 6. Skip surface collision checks if t == 0.
                if (timeMoment == 0.0)
                    continue;

                // 7. Surface collision.
                auto intersection{m_surfaceMesh.getAABBTree().any_intersection(ray)};
                if (!intersection)
                    continue;

                auto triangle{*intersection->second};
                if (triangle.is_degenerate())
                    continue;

#if __cplusplus >= 202002L
                auto matchedIt{std::ranges::find_if(m_surfaceMesh.getTriangleCellMap(), [triangle](auto const &entry)
                                                    { return triangle == entry.second.triangle; })};
#else
                auto matchedIt{std::find_if(m_surfaceMesh.getTriangleCellMap().cbegin(), m_surfaceMesh.getTriangleCellMap().cend(), [triangle](auto const &entry)
                                            { return triangle == entry.second.triangle; })};
#endif

                if (matchedIt != m_surfaceMesh.getTriangleCellMap().end())
                {
                    auto triangleIdOpt{Mesh::isRayIntersectTriangle(ray, matchedIt)};
                    if (triangleIdOpt.has_value())
                    {
                        {
                            std::unique_lock<std::shared_mutex> lock(m_settledParticlesMutex);
                            ++m_settledParticlesCounterMap[triangleIdOpt.value()];
                            m_settledParticlesIds.insert(particle.getId());

                            if (m_settledParticlesIds.size() >= particles.size())
                                notifyStopRequested();
                        }

                        {
                            std::lock_guard<std::mutex> lock(m_particlesMovementMutex);
                            auto intersection_point{RayTriangleIntersection::getIntersectionPoint(ray, triangle)};
                            if (intersection_point)
                                m_particlesMovement[particle.getId()].emplace_back(*intersection_point);
                        }

                        std::cout << "Particle " << particle.getId() << " intersected with " << triangleIdOpt.value() << " triangle\n";
                        if (triangleIdOpt.has_value())
                        {
                            std::unique_lock<std::shared_mutex> lock(m_settledParticlesMutex);
                            ++m_settledParticlesCounterMap[triangleIdOpt.value()];
                            m_settledParticlesIds.insert(particle.getId());

                            if (m_settledParticlesIds.size() >= particles.size())
                                notifyStopRequested();
                        }
                    }
                }
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
        SUCCESSMSG(util::stringify("Time = ", timeMoment, "s. Totally settled: ", m_settledParticlesIds.size(), "/", particles.size(), " particles."));
    }
}
