#ifndef NODECHARGEDENSITYPROCESSOR_HPP
#define NODECHARGEDENSITYPROCESSOR_HPP

#include <mutex>
#include <shared_mutex>

#include "FiniteElementMethod/GSMAssembler.hpp"
#include "Geometry/CubicGrid.hpp"
#include "Particle/Particle.hpp"
#include "Utilities/ConfigParser.hpp"

/**
 * @brief Map of charge densities for nodes in a mesh.
 * @details This type associates each node ID with its corresponding charge density.
 *          The structure is as follows:
 *          - Key: Node ID (GlobalOrdinal)
 *          - Value: Charge density at the node (double)
 */
using NodeChargeDensitiesMap = std::map<GlobalOrdinal, double>;

/**
 * @brief Tracker for particles inside tetrahedrons over time.
 * @details This type organizes particles by simulation time and tetrahedron ID.
 *          The structure is as follows:
 *          - Key: Simulation time (double)
 *          - Value: Map of tetrahedron ID to the particles inside it:
 *            - Key: Tetrahedron ID (size_t)
 *            - Value: Vector of particles inside the tetrahedron (ParticleVector)
 */
using ParticleTrackerMap = std::map<double, std::map<size_t, ParticleVector>>;

/**
 * @brief Set of particle IDs that have settled on a 2D mesh.
 * @details This type stores the IDs of particles that are considered settled
 *          after colliding or interacting with the 2D mesh.
 *          - Element: Particle ID (size_t)
 */
using ParticlesIDSet = std::set<size_t>;

static std::mutex g_nodeChargeDensityMap_mutex;    ///< Mutex for synchronizing access to the charge densities in nodes.
static std::mutex g_particleTrackerMap_mutex;      ///< Mutex for synchronizing access to the particles in tetrahedrons.
static std::shared_mutex g_settledParticles_mutex; ///< Mutex for synchronizing access to settled particle IDs.

/**
 * @brief A utility class for calculating charge densities at nodes in a mesh.
 * @details This class processes particles within a tetrahedral mesh, computes charge densities for tetrahedra,
 *          and aggregates these densities at the nodes of the mesh. It supports both OpenMP-based and standard
 *          multithreaded implementations.
 */
class NodeChargeDensityProcessor
{
public:
    /**
     * @brief Computes the charge densities at nodes of a mesh based on particle positions and charges.
     * @details This function processes particles within a given range and calculates the charge density for
     *          tetrahedra and nodes, using either OpenMP or standard threading, depending on compilation flags.
     * @param timeMoment The current simulation time step.
     * @param configFilename The configuration file name containing simulation settings.
     * @param cubicGrid Shared pointer to a cubic grid instance for spatial indexing.
     * @param gsmAssembler Shared pointer to a GSMAssembler instance for mesh management and volume calculations.
     * @param particles A vector of particles contributing to the charge density.
     * @param settledParticleIDs A set of particle IDs that have settled on a 2D surface mesh.
     * @param particleTrackerMap A map that tracks particles within tetrahedra over time.
     * @param nodeChargeDensityMap A map that stores calculated charge densities for nodes in the mesh.
     * @throw std::runtime_error If an error occurs during particle processing or charge density calculation.
     */
    static void gather(double timeMoment,
                       std::string_view configFilename,
                       std::shared_ptr<CubicGrid> cubicGrid,
                       std::shared_ptr<GSMAssembler> gsmAssembler,
                       ParticleVector const &particles,
                       ParticlesIDSet &settledParticleIDs,
                       ParticleTrackerMap &particleTrackerMap,
                       NodeChargeDensitiesMap &nodeChargeDensityMap);
};

#endif // !NODECHARGEDENSITYPROCESSOR_HPP
