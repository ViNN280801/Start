#ifdef USE_OMP
#include <omp.h>
#endif

#include "Geometry/Utils/Intersections/SegmentTriangleIntersection.hpp"
#include "ParticleInCellEngine/NodeChargeDensityProcessor.hpp"
#include "ParticleInCellEngine/PICExceptions.hpp"
#include "Utilities/ThreadedProcessor.hpp"

#ifndef USE_OMP
/**
 * @brief Helper function for processing charge densities on a range of particles.
 * @details This function handles a specific range of particles, calculates charge densities for tetrahedra
 *          within the range, and aggregates results into global data structures.
 * @param start_index The starting index of the particle range to process.
 * @param end_index The ending index (exclusive) of the particle range to process.
 * @param timeMoment The current simulation time step.
 * @param cubicGrid Shared pointer to a cubic grid instance for spatial indexing.
 * @param gsmAssembler Shared pointer to a GSMAssembler instance for mesh management and volume calculations.
 * @param particles A vector of particles contributing to the charge density.
 * @param settledParticleIDs A set of particle IDs that have settled on a 2D surface mesh.
 * @param particleTrackerMap A map that tracks particles within tetrahedra over time.
 * @param nodeChargeDensityMap A map that stores calculated charge densities for nodes in the mesh.
 */
static void _gather_stdver__helper(
    size_t start_index,
    size_t end_index,
    double timeMoment,
    std::shared_ptr<CubicGrid> cubicGrid,
    std::shared_ptr<GSMAssembler> gsmAssembler,
    ParticleVector const &particles,
    ParticlesIDSet &settledParticleIDs,
    ParticleTrackerMap &particleTrackerMap,
    NodeChargeDensitiesMap &nodeChargeDensityMap)
{
    try
    {
        std::map<size_t, ParticleVector> particleTracker;
        std::map<size_t, double> tetrahedronChargeDensityMap;

        std::for_each(particles.begin() + start_index,
                      particles.begin() + end_index,
                      [&cubicGrid, &particleTracker, &settledParticleIDs](Particle const &particle)
                      {
                          {
                              std::shared_lock<std::shared_mutex> lock(g_settledParticles_mutex);
                              if (settledParticleIDs.find(particle.getId()) != settledParticleIDs.cend())
                                  return;
                          }

                          auto meshParams{cubicGrid->getTetrahedronsByGridIndex(cubicGrid->getGridIndexByPosition(particle.getCentre()))};
                          for (auto const &meshParam : meshParams)
                          {
                              if (meshParam.isPointInside(particle.getCentre()))
                                  particleTracker[meshParam.m_globalTetraId].emplace_back(particle);
                          }
                      });

        // Calculating charge density in each of the tetrahedron using `particleTracker`.
        for (auto const &[globalTetraId, particlesInside] : particleTracker)
            tetrahedronChargeDensityMap.insert({globalTetraId,
                                                (std::accumulate(particlesInside.cbegin(),
                                                                 particlesInside.cend(),
                                                                 0.0,
                                                                 [](double sum, Particle const &particle)
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
                std::lock_guard<std::mutex> lock(g_nodeChargeDensityMap_mutex);
                nodeChargeDensityMap[nodeId] = totalCharge / totalVolume;
            }
        }

        // Adding all the elements from this thread from this local PICTracker to the global PIC tracker.
        std::lock_guard<std::mutex> lock_PIC(g_particleTrackerMap_mutex);
        for (auto const &[tetraId, particlesInside] : particleTracker)
        {
            auto &globalParticles{particleTrackerMap[timeMoment][tetraId]};
            globalParticles.insert(globalParticles.begin(), particlesInside.begin(), particlesInside.end());
        }
    }
    catch (std::exception const &ex)
    {
        START_THROW_EXCEPTION(PICNodeChargeDensityProcessorStdVersionHelperException,
                              util::stringify("Process of the gathering node charge densities failed. Reason: ", ex.what()));
    }
    catch (...)
    {
        START_THROW_EXCEPTION(PICNodeChargeDensityProcessorStdVersionHelperUnknownException,
                              util::stringify("Some error occured while gathering node charge densities"));
    }
}

/**
 * @brief Helper function for processing charge densities using standard threading.
 * @details This function divides particle processing across threads and calculates charge densities
 *          for tetrahedra and nodes in the mesh. Results are stored in shared data structures.
 * @param numThreads Number of threads to use for processing.
 * @param timeMoment The current simulation time step.
 * @param cubicGrid Shared pointer to a cubic grid instance for spatial indexing.
 * @param gsmAssembler Shared pointer to a GSMAssembler instance for mesh management and volume calculations.
 * @param particles A vector of particles contributing to the charge density.
 * @param settledParticleIDs A set of particle IDs that have settled on a 2D surface mesh.
 * @param particleTrackerMap A map that tracks particles within tetrahedra over time.
 * @param nodeChargeDensityMap A map that stores calculated charge densities for nodes in the mesh.
 */
static void _gather_stdver__(
    unsigned int numThreads,
    double timeMoment,
    std::shared_ptr<CubicGrid> cubicGrid,
    std::shared_ptr<GSMAssembler> gsmAssembler,
    ParticleVector const &particles,
    ParticlesIDSet &settledParticleIDs,
    ParticleTrackerMap &particleTrackerMap,
    NodeChargeDensitiesMap &nodeChargeDensityMap)
{
    // We use `std::launch::async` here because calculating charge densities is a compute-intensive task that
    // benefits from immediate parallel execution. Each particle contributes to the overall charge density at
    // different nodes, and tracking them requires processing large numbers of particles across multiple threads.
    // By using `std::launch::async`, we ensure that the processing starts immediately on separate threads,
    // maximizing CPU usage and speeding up the computation to avoid bottlenecks in the simulation.
    ThreadedProcessor::launch(particles.size(), numThreads, std::launch::async,
                              &_gather_stdver__helper,
                              timeMoment,
                              cubicGrid,
                              gsmAssembler,
                              std::cref(particles),
                              std::ref(settledParticleIDs),
                              std::ref(particleTrackerMap),
                              std::ref(nodeChargeDensityMap));
}
#endif // If `USE_OMP` not defined using standard C++ tools to provide concurrency.

#ifdef USE_OMP
/**
 * @brief Helper function for processing charge densities using OpenMP.
 * @details This function uses OpenMP for parallelizing particle processing and charge density calculations.
 *          It divides the work across threads and combines results from per-thread data structures.
 * @param numThreads Number of threads to use for processing.
 * @param timeMoment The current simulation time step.
 * @param cubicGrid Shared pointer to a cubic grid instance for spatial indexing.
 * @param gsmAssembler Shared pointer to a GSMAssembler instance for mesh management and volume calculations.
 * @param particles A vector of particles contributing to the charge density.
 * @param settledParticleIDs A set of particle IDs that have settled on a 2D surface mesh.
 * @param particleTrackerMap A map that tracks particles within tetrahedra over time.
 * @param nodeChargeDensityMap A map that stores calculated charge densities for nodes in the mesh.
 */
static void _gather_ompver__(
    unsigned int numThreads,
    double timeMoment,
    std::shared_ptr<CubicGrid> cubicGrid,
    std::shared_ptr<GSMAssembler> gsmAssembler,
    ParticleVector const &particles,
    ParticlesIDSet &settledParticleIDs,
    ParticleTrackerMap &particleTrackerMap,
    NodeChargeDensitiesMap &nodeChargeDensityMap)
{
    try
    {
        omp_set_num_threads(numThreads);

        std::vector<std::map<size_t, ParticleVector>> particleTracker_per_thread(numThreads);
        std::vector<std::map<size_t, double>> tetrahedronChargeDensityMap_per_thread(numThreads);
        std::vector<std::map<GlobalOrdinal, double>> nodeChargeDensityMap_per_thread(numThreads);

        // First parallel region: process particles
#pragma omp parallel
        {
            int thread_id = omp_get_thread_num();

            // References to per-thread data structures.
            auto &particleTracker = particleTracker_per_thread[thread_id];
            auto &tetrahedronChargeDensityMap = tetrahedronChargeDensityMap_per_thread[thread_id];
            auto &local_nodeChargeDensityMap = nodeChargeDensityMap_per_thread[thread_id];

#pragma omp for schedule(dynamic) nowait
            for (size_t idx = 0ul; idx < particles.size(); ++idx)
            {
                auto const &particle = particles[idx];

                // Check if particle is settled.
                {
                    std::shared_lock<std::shared_mutex> lock(g_settledParticles_mutex);
                    if (settledParticleIDs.find(particle.getId()) != settledParticleIDs.cend())
                        continue;
                }

                auto gridIndex = cubicGrid->getGridIndexByPosition(particle.getCentre());
                auto tetrahedronsData = cubicGrid->getTetrahedronsByGridIndex(gridIndex);

                for (auto const &tetrahedronData : tetrahedronsData)
                {
                    if (tetrahedronData.isPointInside(particle.getCentre()))
                    {
                        // Access per-thread particleTracker.
                        particleTracker[tetrahedronData.m_globalTetraId].emplace_back(particle);
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
            for (size_t i = 0ul; i < nodeTetrahedronsMap.size(); ++i)
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
        for (auto const &pt : particleTracker_per_thread)
        {
            for (auto const &[tetraId, particles] : pt)
            {
                particleTracker[tetraId].insert(particleTracker[tetraId].end(), particles.begin(), particles.end());
            }
        }

        // Merge per-thread tetrahedronChargeDensityMaps into global tetrahedronChargeDensityMap.
        std::map<size_t, double> tetrahedronChargeDensityMap;
        for (auto const &tcdm : tetrahedronChargeDensityMap_per_thread)
        {
            tetrahedronChargeDensityMap.insert(tcdm.begin(), tcdm.end());
        }

        // Merge per-thread nodeChargeDensityMaps into the shared nodeChargeDensityMap.
        {
            std::lock_guard<std::mutex> lock(g_nodeChargeDensityMap_mutex);
            for (auto const &local_map : nodeChargeDensityMap_per_thread)
            {
                for (auto const &[nodeId, chargeDensity] : local_map)
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

        {
            std::lock_guard<std::mutex> lock_PIC(g_particleTrackerMap_mutex);
            for (auto const &[tetraId, particlesInside] : particleTracker)
            {
                auto &globalParticles{particleTrackerMap[timeMoment][tetraId]};
                globalParticles.insert(globalParticles.end(), particlesInside.begin(), particlesInside.end());
            }
        }
    }
    catch (std::exception const &ex)
    {
        START_THROW_EXCEPTION(PICNodeChargeDensityProcessorOmpVersionHelperException,
                              util::stringify("Can't finish PIC processing: ", ex.what()));
    }
    catch (...)
    {
        START_THROW_EXCEPTION(PICNodeChargeDensityProcessorOmpVersionHelperUnknownException,
                              util::stringify("Some error occured while PIC processing"));
    }
}
#endif // !USE_OMP

void NodeChargeDensityProcessor::gather(double timeMoment,
                                        std::string_view configFilename,
                                        std::shared_ptr<CubicGrid> cubicGrid,
                                        std::shared_ptr<GSMAssembler> gsmAssembler,
                                        ParticleVector const &particles,
                                        ParticlesIDSet &settledParticleIDs,
                                        ParticleTrackerMap &particleTrackerMap,
                                        NodeChargeDensitiesMap &nodeChargeDensityMap)
{
    // Check if the requested number of threads exceeds available hardware concurrency.
    ConfigParser configParser(configFilename);
    auto numThreads{configParser.getNumThreads_s()};

#ifdef USE_OMP
    _gather_ompver__(numThreads,
                     timeMoment,
                     cubicGrid,
                     gsmAssembler,
                     particles,
                     settledParticleIDs,
                     particleTrackerMap,
                     nodeChargeDensityMap);
#else
    _gather_stdver__(numThreads,
                     timeMoment,
                     cubicGrid,
                     gsmAssembler,
                     particles,
                     settledParticleIDs,
                     particleTrackerMap,
                     nodeChargeDensityMap);
#endif // !USE_OMP
}
