#include <algorithm>
#include <atomic>
#include <execution>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "DataHandling/HDF5Handler.hpp"
#include "FiniteElementMethod/BoundaryConditions/BoundaryConditionsManager.hpp"
#include "FiniteElementMethod/FEMCheckers.hpp"
#include "FiniteElementMethod/FEMInitializer.hpp"
#include "FiniteElementMethod/FEMLimits.hpp"
#include "FiniteElementMethod/FEMPrinter.hpp"
#include "Generators/ParticleGenerator.hpp"
#include "ModelingMainDriver.hpp"
#include "Particle/CUDA/ParticleDeviceMemoryConverter.cuh"
#include "Particle/PhysicsCore/CollisionModel/CollisionModelFactory.hpp"
#include "ParticleInCellEngine/NodeChargeDensityProcessor.hpp"

std::mutex ModelingMainDriver::m_PICTracker_mutex;
std::mutex ModelingMainDriver::m_nodeChargeDensityMap_mutex;
std::mutex ModelingMainDriver::m_particlesMovement_mutex;
std::shared_mutex ModelingMainDriver::m_settledParticles_mutex;
std::atomic_flag ModelingMainDriver::m_stop_processing = ATOMIC_FLAG_INIT;

void ModelingMainDriver::_broadcastTriangleMesh()
{
    int rank{};
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, std::addressof(rank));
#endif

    size_t numTriangles{};
    if (rank == 0)
        numTriangles = _triangleMesh.size();

#ifdef USE_MPI
    // Broadcast the number of triangles.
    MPI_Bcast(std::addressof(numTriangles), 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
#endif

    // Prepare arrays for sending or receiving.
    std::vector<size_t> triangleIds(numTriangles);
    std::vector<double> triangleCoords(numTriangles * 9); // 9 doubles per triangle.
    std::vector<double> dS_values(numTriangles);
    std::vector<int> counters(numTriangles);

    if (rank == 0)
    {
        // Prepare data on rank 0.
        for (size_t i{}; i < numTriangles; ++i)
        {
            auto const &meshParam{_triangleMesh[i]};
            triangleIds[i] = std::get<0>(meshParam);
            Triangle const &triangle{std::get<1>(meshParam)};
            double dS{std::get<2>(meshParam)};
            int counter{std::get<3>(meshParam)};

            // Extract triangle coordinates
            for (int j{}; j < 3; ++j)
            {
                const Point &p = triangle.vertex(j);
                triangleCoords[i * 9 + j * 3 + 0] = p.x();
                triangleCoords[i * 9 + j * 3 + 1] = p.y();
                triangleCoords[i * 9 + j * 3 + 2] = p.z();
            }
            dS_values[i] = dS;
            counters[i] = counter;
        }
    }

#ifdef USE_MPI
    // Broadcast the data arrays.
    MPI_Bcast(triangleIds.data(), numTriangles, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(triangleCoords.data(), numTriangles * 9, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(dS_values.data(), numTriangles, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(counters.data(), numTriangles, MPI_INT, 0, MPI_COMM_WORLD);
#endif

    if (rank != 0)
    {
        // Reconstruct the triangle mesh on other ranks
        _triangleMesh.clear();
        for (size_t i = 0; i < numTriangles; ++i)
        {
            size_t triangleId = triangleIds[i];
            Point p1(triangleCoords[i * 9 + 0], triangleCoords[i * 9 + 1], triangleCoords[i * 9 + 2]);
            Point p2(triangleCoords[i * 9 + 3], triangleCoords[i * 9 + 4], triangleCoords[i * 9 + 5]);
            Point p3(triangleCoords[i * 9 + 6], triangleCoords[i * 9 + 7], triangleCoords[i * 9 + 8]);
            Triangle triangle(p1, p2, p3);
            double dS{dS_values[i]};
            int counter{counters[i]};
            _triangleMesh.emplace_back(std::make_tuple(triangleId, triangle, dS, counter));
        }
    }
}

void ModelingMainDriver::_initializeSurfaceMesh()
{
    int rank{};
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, std::addressof(rank));
#endif

    if (rank == 0)
        _triangleMesh = Mesh::getMeshParams(m_config.getMeshFilename());

    _broadcastTriangleMesh();
}

void ModelingMainDriver::_initializeSurfaceMeshAABB()
{
    if (_triangleMesh.empty())
        throw std::runtime_error("Can't construct AABB for triangle mesh - surface mesh is empty");

    for (auto const &meshParam : _triangleMesh)
    {
        auto const &triangle{std::get<1>(meshParam)};
        if (!triangle.is_degenerate())
            _triangles.emplace_back(triangle);
    }

    if (_triangles.empty())
        throw std::runtime_error("Can't create AABB for triangle mesh - triangles from the mesh are invalid. Possible reason: all the triangles are degenerate");

    _surfaceMeshAABBtree = AABB_Tree_Triangle(std::cbegin(_triangles), std::cend(_triangles));
}

void ModelingMainDriver::_initializeParticles()
{
    if (m_config.isParticleSourcePoint())
    {
        auto tmp{ParticleGenerator::fromPointSource(m_config.getParticleSourcePoints())};
        ParticleVector h_particles{ParticleDeviceMemoryConverter::copyToHost(tmp)};
        if (!h_particles.empty())
            m_particles.insert(m_particles.end(), std::begin(h_particles), std::end(h_particles));
    }
    if (m_config.isParticleSourceSurface())
    {
        auto tmp{ParticleGenerator::fromSurfaceSource(m_config.getParticleSourceSurfaces())};
        ParticleVector h_particles{ParticleDeviceMemoryConverter::copyToHost(tmp)};
        if (!h_particles.empty())
            m_particles.insert(m_particles.end(), std::begin(h_particles), std::end(h_particles));
    }

    if (m_particles.empty())
        throw std::runtime_error("Particles are uninitialized, check your configuration file");
}

void ModelingMainDriver::_ginitialize()
{
    _initializeSurfaceMesh();
    _initializeSurfaceMeshAABB();
    _initializeParticles();
}

void ModelingMainDriver::_updateSurfaceMesh()
{
    // Updating hdf5file to know how many particles settled on certain triangle from the surface mesh.
    auto mapEnd{_settledParticlesCounterMap.cend()};
    for (auto &meshParam : _triangleMesh)
        if (auto it{_settledParticlesCounterMap.find(std::get<0>(meshParam))}; it != mapEnd)
            std::get<3>(meshParam) = it->second;

    std::string hdf5filename(std::string(m_config.getMeshFilename().substr(0ul, m_config.getMeshFilename().find("."))));
    hdf5filename += ".hdf5";
    HDF5Handler hdf5handler(hdf5filename);
    hdf5handler.saveMeshToHDF5(_triangleMesh);
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

template <typename Function, typename... Args>
void ModelingMainDriver::_processWithThreads(unsigned int num_threads, Function &&function, std::launch launch_policy, Args &&...args)
{
    // Static assert to ensure the number of threads is greater than 0.
    static_assert(sizeof...(args) > 0, "You must provide at least one argument to pass to the function.");

    // Check if the requested number of threads exceeds available hardware concurrency.
    unsigned int available_threads{std::thread::hardware_concurrency()};
    if (num_threads > available_threads)
        throw std::invalid_argument("The number of threads requested exceeds the available hardware threads.");

    // Handle edge case of num_threads == 0.
    if (num_threads == 0)
        throw std::invalid_argument("The number of threads must be greater than 0.");

    size_t particles_per_thread{m_particles.size() / num_threads}, start_index{},
        managed_particles{particles_per_thread * num_threads}; // Count of the managed particles.

    std::vector<std::future<void>> futures;

    // Create threads and assign each a segment of particles to process.
    for (size_t i{}; i < num_threads; ++i)
    {
        size_t end_index{(i == num_threads - 1) ? m_particles.size() : start_index + particles_per_thread};

        // If, for example, the simulation started with 1,000 particles and 18 threads, we would lose 10 particles: 1000/18=55 => 55*18=990.
        // In this case, we assign all remaining particles to the last thread to manage them.
        if (i == num_threads - 1 && managed_particles < m_particles.size())
            end_index = m_particles.size();

        // Launch the function asynchronously for the current segment.
        futures.emplace_back(std::async(launch_policy, [this, start_index, end_index, &function, &args...]()
                                        { std::invoke(function, this, start_index, end_index, std::forward<Args>(args)...); }));
        start_index = end_index;
    }

    // Wait for all threads to complete their work.
    for (auto &f : futures)
        f.get();
}

ModelingMainDriver::ModelingMainDriver(std::string_view config_filename) : m_config(config_filename), __expt_m_config_filename(config_filename)
{
    // Checking mesh filename on validity and assign it to the class member.
    FEMCheckers::checkMeshFile(m_config.getMeshFilename());

    // Calculating and checking gas concentration.
    _gasConcentration = util::calculateConcentration(config_filename);
    if (_gasConcentration < constants::gasConcentrationMinimalValue)
    {
        WARNINGMSG(util::stringify("Something wrong with the concentration of the gas. Its value is ", _gasConcentration, ". Simulation might considerably slows down"));
    }

    // Global initializator. Initializes surface mesh, AABB for this mesh and spawning particles.
    _ginitialize();
}

ModelingMainDriver::~ModelingMainDriver() { _gfinalize(); }

void ModelingMainDriver::_processParticleTracker(size_t start_index, size_t end_index, double t,
                                                 std::shared_ptr<CubicGrid> cubicGrid, std::shared_ptr<GSMAssembler> gsmAssembler,
                                                 std::map<GlobalOrdinal, double> &nodeChargeDensityMap)
{
    try
    {
#ifdef USE_OMP
        // Check if the requested number of threads exceeds available hardware concurrency.
        auto num_threads{m_config.getNumThreads_s()};

        omp_set_num_threads(num_threads);

        std::vector<std::map<size_t, ParticleVector>> particleTracker_per_thread(num_threads);
        std::vector<std::map<size_t, double>> tetrahedronChargeDensityMap_per_thread(num_threads);
        std::vector<std::map<GlobalOrdinal, double>> nodeChargeDensityMap_per_thread(num_threads);

        // First parallel region: process particles
#pragma omp parallel
        {
            int thread_id = omp_get_thread_num();

            // References to per-thread data structures.
            auto &particleTracker = particleTracker_per_thread[thread_id];
            auto &tetrahedronChargeDensityMap = tetrahedronChargeDensityMap_per_thread[thread_id];
            auto &local_nodeChargeDensityMap = nodeChargeDensityMap_per_thread[thread_id];

#pragma omp for schedule(dynamic) nowait
            for (size_t idx = start_index; idx < end_index; ++idx)
            {
                auto const &particle = m_particles[idx];

                // Check if particle is settled.
                {
                    std::shared_lock<std::shared_mutex> lock(m_settledParticles_mutex);
                    if (_settledParticlesIds.find(particle.getId()) != _settledParticlesIds.cend())
                        continue;
                }

                auto gridIndex = cubicGrid->getGridIndexByPosition(particle.getCentre());
                auto meshParams = cubicGrid->getTetrahedronsByGridIndex(gridIndex);

                for (auto const &meshParam : meshParams)
                {
                    if (Mesh::isPointInsideTetrahedron(particle.getCentre(), meshParam.tetrahedron))
                    {
                        // Access per-thread particleTracker.
                        particleTracker[meshParam.globalTetraId].emplace_back(particle);
                    }
                }
            } // end of particle loop.

            // Calculate charge density in each tetrahedron for this thread.
            for (auto const &[globalTetraId, particlesInside] : particleTracker)
            {
                double totalCharge = std::accumulate(particlesInside.cbegin(), particlesInside.cend(), 0.0,
                                                     [](double sum, Particle const &particle)
                                                     { return sum + particle.getCharge(); });
                double volume = gsmAssembler->getMeshManager().getVolumeByGlobalTetraId(globalTetraId);
                double chargeDensity = totalCharge / volume;
                tetrahedronChargeDensityMap[globalTetraId] = chargeDensity;
            }

            // Access node-tetrahedron mapping.
            auto nodeTetrahedronsMap = gsmAssembler->getMeshManager().getNodeTetrahedronsMap();

            // Process nodes and aggregate data from adjacent tetrahedra.
#pragma omp for schedule(dynamic) nowait
            for (size_t i = 0; i < nodeTetrahedronsMap.size(); ++i)
            {
                auto it = nodeTetrahedronsMap.begin();
                std::advance(it, i);
                auto const &nodeId = it->first;
                auto const &adjacentTetrahedra = it->second;

                double totalCharge = 0.0;
                double totalVolume = 0.0;

                // Sum up the charge and volume for all tetrahedra of a given node.
                for (auto const &tetrId : adjacentTetrahedra)
                {
                    auto tcd_it = tetrahedronChargeDensityMap.find(tetrId);
                    if (tcd_it != tetrahedronChargeDensityMap.end())
                    {
                        double tetrahedronChargeDensity = tcd_it->second;
                        double tetrahedronVolume = gsmAssembler->getMeshManager().getVolumeByGlobalTetraId(tetrId);

                        totalCharge += tetrahedronChargeDensity * tetrahedronVolume;
                        totalVolume += tetrahedronVolume;
                    }
                }

                // Calculate and store the charge density for the node.
                if (totalVolume > 0)
                {
                    double nodeChargeDensity = totalCharge / totalVolume;
                    local_nodeChargeDensityMap[nodeId] = nodeChargeDensity;
                }
            }
        } // end of parallel region.

        // Merge per-thread particleTrackers into global particleTracker.
        std::map<size_t, ParticleVector> particleTracker;
        for (const auto &pt : particleTracker_per_thread)
        {
            for (const auto &[tetraId, particles] : pt)
            {
                particleTracker[tetraId].insert(particleTracker[tetraId].end(), particles.begin(), particles.end());
            }
        }

        // Merge per-thread tetrahedronChargeDensityMaps into global tetrahedronChargeDensityMap.
        std::map<size_t, double> tetrahedronChargeDensityMap;
        for (const auto &tcdm : tetrahedronChargeDensityMap_per_thread)
        {
            tetrahedronChargeDensityMap.insert(tcdm.begin(), tcdm.end());
        }

        // Merge per-thread nodeChargeDensityMaps into the shared nodeChargeDensityMap.
        {
            std::lock_guard<std::mutex> lock(m_nodeChargeDensityMap_mutex);
            for (const auto &local_map : nodeChargeDensityMap_per_thread)
            {
                for (const auto &[nodeId, chargeDensity] : local_map)
                {
                    // If the node already exists, average the charge densities.
                    auto it = nodeChargeDensityMap.find(nodeId);
                    if (it != nodeChargeDensityMap.end())
                    {
                        it->second = (it->second + chargeDensity) / 2.0;
                    }
                    else
                    {
                        nodeChargeDensityMap[nodeId] = chargeDensity;
                    }
                }
            }
        }

        // Add the particles from particleTracker to m_particleTracker[t].
        {
            std::lock_guard<std::mutex> lock_PIC(m_PICTracker_mutex);
            for (auto const &[tetraId, particlesInside] : particleTracker)
            {
                auto &globalParticles{m_particleTracker[t][tetraId]};
                globalParticles.insert(globalParticles.end(), particlesInside.begin(), particlesInside.end());
            }
        }
#else
        std::map<size_t, ParticleVector> particleTracker;
        std::map<size_t, double> tetrahedronChargeDensityMap;

        std::for_each(m_particles.begin() + start_index, m_particles.begin() + end_index, [this, &cubicGrid, &particleTracker](auto &particle)
                      {
        if (_settledParticlesIds.find(particle.getId()) != _settledParticlesIds.cend())
            return;

        auto meshParams{cubicGrid->getTetrahedronsByGridIndex(cubicGrid->getGridIndexByPosition(particle.getCentre()))};
        for (auto const &meshParam : meshParams)
            if (Mesh::isPointInsideTetrahedron(particle.getCentre(), meshParam.tetrahedron))
                particleTracker[meshParam.globalTetraId].emplace_back(particle); });

        // Calculating charge density in each of the tetrahedron using `particleTracker`.
        for (auto const &[globalTetraId, particlesInside] : particleTracker)
            tetrahedronChargeDensityMap.insert({globalTetraId,
                                                (std::accumulate(particlesInside.cbegin(), particlesInside.cend(), 0.0, [](double sum, Particle const &particle)
                                                                 { return sum + particle.getCharge(); })) /
                                                    gsmAssembler->getMeshManager().getVolumeByGlobalTetraId(globalTetraId)});

        // Go around each node and aggregate data from adjacent tetrahedra.
        for (auto const &[nodeId, adjecentTetrahedrons] : gsmAssembler->getMeshManager().getNodeTetrahedronsMap())
        {
            double totalCharge{}, totalVolume{};

            // Sum up the charge and volume for all tetrahedra of a given node.
            for (auto const &tetrId : adjecentTetrahedrons)
            {
                if (tetrahedronChargeDensityMap.find(tetrId) != tetrahedronChargeDensityMap.end())
                {
                    double tetrahedronChargeDensity{tetrahedronChargeDensityMap.at(tetrId)},
                        tetrahedronVolume{gsmAssembler->getMeshManager().getVolumeByGlobalTetraId(tetrId)};

                    totalCharge += tetrahedronChargeDensity * tetrahedronVolume;
                    totalVolume += tetrahedronVolume;
                }
            }

            // Calculate and store the charge density for the node.
            if (totalVolume > 0)
            {
                std::lock_guard<std::mutex> lock(m_nodeChargeDensityMap_mutex);
                nodeChargeDensityMap[nodeId] = totalCharge / totalVolume;
            }
        }

        // Adding all the elements from this thread from this local PICTracker to the global PIC tracker.
        std::lock_guard<std::mutex> lock_PIC(m_PICTracker_mutex);
        for (auto const &[tetraId, particlesInside] : particleTracker)
        {
            auto &globalParticles{m_particleTracker[t][tetraId]};
            globalParticles.insert(globalParticles.begin(), particlesInside.begin(), particlesInside.end());
        }
#endif // !USE_OMP
    }
    catch (std::exception const &ex)
    {
        ERRMSG(util::stringify("Can't finish PIC processing: ", ex.what()));
    }
    catch (...)
    {
        ERRMSG("Some error occured while PIC processing");
    }
}

void ModelingMainDriver::_solveEquation(std::map<GlobalOrdinal, double> &nodeChargeDensityMap,
                                        std::shared_ptr<GSMAssembler> &gsmAssembler,
                                        std::shared_ptr<VectorManager> &solutionVector,
                                        std::map<GlobalOrdinal, double> &boundaryConditions, double time)
{
    try
    {
        auto nonChangebleNodes{m_config.getNonChangeableNodes()};
        for (auto const &[nodeId, nodeChargeDensity] : nodeChargeDensityMap)
#if __cplusplus >= 202002L
            if (std::ranges::find(nonChangebleNodes, nodeId) == nonChangebleNodes.cend())
#else
            if (std::find(nonChangebleNodes.cbegin(), nonChangebleNodes.cend(), nodeId) == nonChangebleNodes.cend())
#endif
                boundaryConditions[nodeId] = nodeChargeDensity;
        BoundaryConditionsManager::set(solutionVector->get(), FEM_LIMITS_DEFAULT_POLYNOMIAL_ORDER, boundaryConditions);

        MatrixEquationSolver solver(gsmAssembler, solutionVector);
        auto solverParams{solver.createSolverParams(m_config.getSolverName(), m_config.getMaxIterations(), m_config.getConvergenceTolerance(),
                                                    m_config.getVerbosity(), m_config.getOutputFrequency(), m_config.getNumBlocks(), m_config.getBlockSize(),
                                                    m_config.getMaxRestarts(), m_config.getFlexibleGMRES(), m_config.getOrthogonalization(),
                                                    m_config.getAdaptiveBlockSize(), m_config.getConvergenceTestFrequency())};
        solver.solve(m_config.getSolverName(), solverParams);
        solver.calculateElectricField(); // Getting electric field for the each cell.

        solver.writeElectricPotentialsToPosFile(time);
        solver.writeElectricFieldVectorsToPosFile(time);
    }
    catch (std::exception const &ex)
    {
        ERRMSG(util::stringify("Can't solve the equation: ", ex.what()));
    }
    catch (...)
    {
        ERRMSG("Some error occured while solving the matrix equation Ax=b");
    }
}

void ModelingMainDriver::_processPIC_and_SurfaceCollisionTracker(size_t start_index, size_t end_index, double t,
                                                                 std::shared_ptr<CubicGrid> cubicGrid, std::shared_ptr<GSMAssembler> gsmAssembler)
{
    try
    {
#ifdef USE_OMP
        // OpenMP implementation.
        MagneticInduction magneticInduction{}; // Assuming induction vector B is 0.

#pragma omp parallel
        {
            // Each thread will process a subset of particles.
#pragma omp for
            for (size_t idx = start_index; idx < end_index; ++idx)
            {
                auto &particle = m_particles[idx];

                // Check if particle is already settled.
                bool isSettled = false;
                {
                    std::shared_lock<std::shared_mutex> lock(m_settledParticles_mutex);
                    if (_settledParticlesIds.find(particle.getId()) != _settledParticlesIds.end())
                        isSettled = true;
                }
                if (isSettled)
                    continue;

                size_t containingTetrahedron = 0;

                {
                    // Access m_particleTracker[t] safely.
                    std::lock_guard<std::mutex> lock(m_PICTracker_mutex);
                    auto it = m_particleTracker.find(t);
                    if (it != m_particleTracker.end())
                    {
                        auto &particleMap = it->second;
                        for (auto const &[tetraId, particlesInside] : particleMap)
                        {
                            if (std::any_of(particlesInside.cbegin(), particlesInside.cend(),
                                            [&particle](Particle const &storedParticle)
                                            { return particle.getId() == storedParticle.getId(); }))
                            {
                                containingTetrahedron = tetraId;
                                break;
                            }
                        }
                    }
                }

                if (auto tetrahedron = gsmAssembler->getMeshManager().getMeshDataByTetrahedronId(containingTetrahedron))
                {
                    if (tetrahedron->electricField.has_value())
                    {
                        // Updating velocity of the particle according to the Lorentz force.
                        particle.electroMagneticPush(
                            magneticInduction,
                            ElectricField(tetrahedron->electricField->x(), tetrahedron->electricField->y(), tetrahedron->electricField->z()),
                            m_config.getTimeStep());
                    }
                }

                Point prev(particle.getCentre());

                {
                    // Lock m_particlesMovement_mutex when modifying m_particlesMovement.
                    std::lock_guard<std::mutex> lock(m_particlesMovement_mutex);

                    // Adding only those particles which are inside tetrahedron mesh.
                    if (cubicGrid->isInsideTetrahedronMesh(prev) && m_particlesMovement.size() <= kdefault_max_numparticles_to_anim)
                        m_particlesMovement[particle.getId()].emplace_back(prev);
                }

                particle.updatePosition(m_config.getTimeStep());
                Ray ray(prev, particle.getCentre());

                if (ray.is_degenerate())
                    continue;

                // Updating velocity of the particle according to the collision with gas.
                auto collisionModel = CollisionModelFactory::create(m_config.getScatteringModel());
                bool collided = collisionModel->collide(particle, m_config.getGas(), _gasConcentration, m_config.getTimeStep());

                // Skip collision detection at initial time.
                if (t == 0.0)
                    continue;

                auto intersection = _surfaceMeshAABBtree.any_intersection(ray);
                if (!intersection)
                    continue;

                auto triangle = *intersection->second;
                if (triangle.is_degenerate())
                    continue;

#if __cplusplus >= 202002L
                auto matchedIt = std::ranges::find_if(_triangleMesh, [triangle](auto const &el)
                                                      { return triangle == std::get<1>(el); });
#else
                auto matchedIt = std::find_if(_triangleMesh.cbegin(), _triangleMesh.cend(), [triangle](auto const &el)
                                              { return triangle == std::get<1>(el); });
#endif

                if (matchedIt != _triangleMesh.cend())
                {
                    auto id = Mesh::isRayIntersectTriangle(ray, *matchedIt);
                    if (id)
                    {
                        bool stopProcessing = false;
                        {
                            std::unique_lock<std::shared_mutex> lock(m_settledParticles_mutex);
                            ++_settledParticlesCounterMap[id.value()];
                            _settledParticlesIds.insert(particle.getId());

                            if (_settledParticlesIds.size() >= m_particles.size())
                            {
                                m_stop_processing.test_and_set();
                                stopProcessing = true;
                            }
                        }

                        {
                            std::lock_guard<std::mutex> lock(m_particlesMovement_mutex);
                            auto intersection_point = RayTriangleIntersection::getIntersectionPoint(ray, triangle);
                            if (intersection_point)
                                m_particlesMovement[particle.getId()].emplace_back(*intersection_point);
                        }

                        if (stopProcessing)
                            continue;
                    }
                }
            } // end of for loop.
        }     // end of parallel region.
#else
        MagneticInduction magneticInduction{}; // For brevity assuming that induction vector B is 0.
        std::for_each(m_particles.begin() + start_index, m_particles.begin() + end_index,
                      [this, &cubicGrid, gsmAssembler, magneticInduction, t](auto &particle)
                      {
                          {
                              // If particles is already settled on surface - there is no need to proceed handling it.
                              std::shared_lock<std::shared_mutex> lock(m_settledParticles_mutex);
                              if (_settledParticlesIds.find(particle.getId()) != _settledParticlesIds.end())
                                  return;
                          }

                          size_t containingTetrahedron{};
                          for (auto const &[tetraId, particlesInside] : m_particleTracker[t])
                          {
#if __cplusplus >= 202002L
                              if (std::ranges::find_if(particlesInside, [particle](Particle const &storedParticle)
                                                       { return particle.getId() == storedParticle.getId(); }) != particlesInside.cend())
#else
                    if (std::find_if(particlesInside, [particle](Particle const &storedParticle)
                                     { return particle.getId() == storedParticle.getId(); }) != particlesInside.cend())
#endif
                              {
                                  containingTetrahedron = tetraId;
                                  break;
                              }
                          }

                          if (auto tetrahedron{gsmAssembler->getMeshManager().getMeshDataByTetrahedronId(containingTetrahedron)})
                              if (tetrahedron->electricField.has_value())
                                  // Updating velocity of the particle according to the Lorentz force.
                                  particle.electroMagneticPush(magneticInduction,
                                                               ElectricField(tetrahedron->electricField->x(), tetrahedron->electricField->y(), tetrahedron->electricField->z()),
                                                               m_config.getTimeStep());

                          Point prev(particle.getCentre());

                          {
                              std::lock_guard<std::mutex> lock(m_particlesMovement_mutex);

                              // Adding only those particles which are inside tetrahedron mesh.
                              // There is no need to spawn large count of particles and load PC, fixed count must be enough.
                              if (cubicGrid->isInsideTetrahedronMesh(prev) && m_particlesMovement.size() <= kdefault_max_numparticles_to_anim)
                                  m_particlesMovement[particle.getId()].emplace_back(prev);
                          }

                          particle.updatePosition(m_config.getTimeStep());
                          Ray ray(prev, particle.getCentre());

                          if (ray.is_degenerate())
                              return;

                          // Updating velocity of the particle according to the coliding with gas.
                          auto collisionModel = CollisionModelFactory::create(m_config.getScatteringModel());
                          bool collided = collisionModel->collide(particle, m_config.getGas(), _gasConcentration, m_config.getTimeStep());

                          // There is no need to check particle collision with surface mesh in initial time moment of the simulation (when t = 0).
                          if (t == 0.0)
                              return;

                          auto intersection{_surfaceMeshAABBtree.any_intersection(ray)};
                          if (!intersection)
                              return;

                          auto triangle{*intersection->second};
                          if (triangle.is_degenerate())
                              return;

#if __cplusplus >= 202002L
                          auto matchedIt{std::ranges::find_if(_triangleMesh, [triangle](auto const &el)
                                                              { return triangle == std::get<1>(el); })};
#else
                auto matchedIt{std::find_if(_triangleMesh, [triangle](auto const &el)
                                            { return triangle == std::get<1>(el); })};
#endif
                          if (matchedIt != _triangleMesh.cend())
                          {
                              auto id{Mesh::isRayIntersectTriangle(ray, *matchedIt)};
                              if (id)
                              {
                                  {
                                      std::shared_lock<std::shared_mutex> lock(m_settledParticles_mutex);
                                      ++_settledParticlesCounterMap[id.value()];
                                      _settledParticlesIds.insert(particle.getId());

                                      if (_settledParticlesIds.size() >= m_particles.size())
                                      {
                                          m_stop_processing.test_and_set();
                                          return;
                                      }
                                  }

                                  {
                                      std::lock_guard<std::mutex> lock(m_particlesMovement_mutex);
                                      auto intersection_point{RayTriangleIntersection::getIntersectionPoint(ray, triangle)};
                                      if (intersection_point)
                                          m_particlesMovement[particle.getId()].emplace_back(*intersection_point);
                                  }
                              }
                          }
                      });
#endif // !USE_OMP
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

void ModelingMainDriver::startModeling()
{
    /* Beginning of the FEM initialization. */
    FEMInitializer feminit(m_config);
    auto gsmAssembler{feminit.getGlobalStiffnessMatrixAssembler()};
    auto cubicGrid{feminit.getCubicGrid()};
    auto boundaryConditions{feminit.getBoundaryConditions()};
    auto solutionVector{feminit.getEquationRHS()};
    /* Ending of the FEM initialization. */

    [[maybe_unused]] auto num_threads{m_config.getNumThreads_s()};
    std::map<GlobalOrdinal, double> nodeChargeDensityMap;

#if __cplusplus >= 202002L
    for (double t{}; t <= m_config.getSimulationTime() && !m_stop_processing.test(); t += m_config.getTimeStep())
#else
    for (double t{}; t <= m_config.getSimulationTime() && !m_stop_processing.test_and_set(); t += m_config.getTimeStep())
#endif
    {
#if __cplusplus < 202002L
        // Clear the flag immediately so it doesn't stay set.
        m_stop_processing.clear();
#endif

        NodeChargeDensityProcessor::gather(t,
                                           __expt_m_config_filename,
                                           cubicGrid,
                                           gsmAssembler,
                                           m_particles,
                                           _settledParticlesIds,
                                           m_particleTracker,
                                           nodeChargeDensityMap);

        // 2. Solve equation in the main thread.
        _solveEquation(nodeChargeDensityMap, gsmAssembler, solutionVector, boundaryConditions, t);

        // 3. Process surface collision tracking in parallel.
#ifdef USE_OMP
        _processPIC_and_SurfaceCollisionTracker(0, m_particles.size(), t, cubicGrid, gsmAssembler);
#else
        // We use `std::launch::deferred` here because surface collision tracking is not critical to be run immediately
        // after particle tracking. It can be deferred until it is needed, allowing for potential optimizations
        // such as saving resources when not required right away. Deferring this task ensures that the function
        // only runs when explicitly requested via `get()` or `wait()`, thereby reducing overhead if the results are
        // not immediately needed. This approach also helps avoid contention with more urgent tasks running in parallel.
        _processWithThreads(num_threads, &ModelingMainDriver::_processPIC_and_SurfaceCollisionTracker, std::launch::deferred, t, cubicGrid, gsmAssembler);
#endif
    }
    LOGMSG(util::stringify("Totally settled: ", _settledParticlesIds.size(), "/", m_particles.size(), " particles."));
}
