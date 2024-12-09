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

ModelingMainDriver::ModelingMainDriver(std::string_view config_filename)
    : m_config_filename(config_filename),
      m_config(config_filename)
{
    // Checking mesh filename on validity and assign it to the class member.
    FEMCheckers::checkMeshFile(m_config.getMeshFilename());

    // Calculating and checking gas concentration.
    _gasConcentration = util::calculateConcentration_w(config_filename);

    // Global initializator. Initializes surface mesh, AABB for this mesh and spawning particles.
    _ginitialize();
}

ModelingMainDriver::~ModelingMainDriver() { _gfinalize(); }

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

    for (double timeMoment{}; timeMoment <= m_config.getSimulationTime(); timeMoment += m_config.getTimeStep())
    {
        NodeChargeDensityProcessor::gather(timeMoment,
                                           m_config_filename,
                                           cubicGrid,
                                           gsmAssembler,
                                           m_particles,
                                           _settledParticlesIds,
                                           m_particleTracker,
                                           nodeChargeDensityMap);

        // 2. Solve equation in the main thread.
        ChargeDensityEquationSolver::solve(timeMoment,
                                           m_config_filename,
                                           nodeChargeDensityMap,
                                           gsmAssembler,
                                           solutionVector,
                                           boundaryConditions);

        // 3. Process surface collision tracking in parallel.
        ParticleDynamicsProcessor particleDynamicProcessor(m_config_filename, m_settledParticlesMutex, m_particlesMovementMutex,
                                                           _surfaceMeshAABBtree, _triangleMesh, _settledParticlesIds,
                                                           _settledParticlesCounterMap, m_particlesMovement);
        particleDynamicProcessor.process(m_config_filename, m_particles, timeMoment, cubicGrid, gsmAssembler, m_particleTracker);
    }
    LOGMSG(util::stringify("Totally settled: ", _settledParticlesIds.size(), "/", m_particles.size(), " particles."));
}
