#include <hdf5.h>
#include <mpi.h>
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

void _readParticlePositionsHdf5(const std::string &filepath = "results/particles_movements.hdf5", int particleID = -1)
{
    try
    {
        // Open the HDF5 file
        hid_t file_id = H5Fopen(filepath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file_id < 0)
        {
            ERRMSG("Failed to open HDF5 file: " + filepath);
            return;
        }

        // Get the list of all datasets in the file
        hsize_t num_objs;
        H5Gget_num_objs(file_id, &num_objs);

        bool particleFound = false;

        for (hsize_t i = 0; i < num_objs; ++i)
        {
            char dataset_name[1024];
            H5Gget_objname_by_idx(file_id, i, dataset_name, sizeof(dataset_name));

            // If particleID is -1, print all particles
            if (particleID == -1 || dataset_name == ("Particle_" + std::to_string(particleID)))
            {
                particleFound = true;

                // Open the dataset
                hid_t dataset_id = H5Dopen(file_id, dataset_name, H5P_DEFAULT);
                if (dataset_id < 0)
                {
                    ERRMSG("Failed to open dataset: " + std::string(dataset_name));
                    continue;
                }

                // Get the dataspace and dimensions
                hid_t dataspace_id = H5Dget_space(dataset_id);
                hsize_t dims[2];
                H5Sget_simple_extent_dims(dataspace_id, dims, NULL);

                // Read the data
                std::vector<double> positions(dims[0] * dims[1]);
                H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, positions.data());

                // Print the particle's positions
                std::cout << dataset_name << ":\n";
                for (hsize_t j = 0; j < dims[0]; ++j)
                {
                    std::cout << "  " << j + 1 << ". [" << positions[j * 3] << ", "
                              << positions[j * 3 + 1] << ", " << positions[j * 3 + 2] << "]\n";
                }

                // Close the dataset and dataspace
                H5Dclose(dataset_id);
                H5Sclose(dataspace_id);

                // If a specific particle ID was requested, exit the loop after finding it
                if (particleID != -1)
                    break;
            }
        }

        if (!particleFound)
        {
            if (particleID == -1)
            {
                ERRMSG("No particles found in the file.");
            }
            else
            {
                ERRMSG("Particle with ID " + std::to_string(particleID) + " not found.");
            }
        }

        // Close the file
        H5Fclose(file_id);
    }
    catch (...)
    {
        ERRMSG("An error occurred while reading particle positions from HDF5.");
    }
}

void _saveParticleMovementsHdf5(ParticleMovementMap const &particlesMovement)
{
    try
    {
        if (particlesMovement.empty())
        {
            WARNINGMSG("Warning: Particle movements map is empty, no data to save");
            return;
        }

        // Create or open the HDF5 file
        std::string filepath("results/particles_movements.hdf5");
        hid_t file_id = H5Fcreate(filepath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

        for (auto const &[id, movements] : particlesMovement)
        {
            if (movements.empty())
                continue;

            // Prepare the data for this particle
            std::vector<double> positions;
            for (auto const &point : movements)
            {
                positions.emplace_back(point.x());
                positions.emplace_back(point.y());
                positions.emplace_back(point.z());
            }

            // Create dataspace
            hsize_t dims[2] = {movements.size(), 3}; // Nx3 array
            hid_t dataspace_id = H5Screate_simple(2, dims, NULL);

            // Create dataset
            std::string dataset_name = "Particle_" + std::to_string(id);
            hid_t dataset_id = H5Dcreate2(file_id, dataset_name.c_str(), H5T_NATIVE_DOUBLE, dataspace_id,
                                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

            // Write data to the dataset
            H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, positions.data());

            // Close dataset and dataspace
            H5Dclose(dataset_id);
            H5Sclose(dataspace_id);
        }

        // Close the file
        H5Fclose(file_id);

        LOGMSG("Successfully written particle movements to the HDF5 file.");
    }
    catch (...)
    {
        ERRMSG("An error occurred while saving particle movements to HDF5.");
    }
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
    _saveParticleMovementsHdf5(m_particlesMovement);
    _readParticlePositionsHdf5();
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

    ParticleSurfaceCollisionHandler surfaceCollisionHandler(m_settledParticlesMutex, m_particlesMovementMutex, m_surfaceMesh.getAABBTree(),
                                                            m_surfaceMesh.getTriangleCellMap(), m_settledParticlesIds, m_settledParticlesCounterMap, m_particlesMovement, *this);

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
                surfaceCollisionHandler.handle(particle, ray, particles.size());
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
