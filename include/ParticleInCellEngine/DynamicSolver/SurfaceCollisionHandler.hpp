#ifndef SURFACECOLLISIONHANDLER_HPP
#define SURFACECOLLISIONHANDLER_HPP

#include <optional>
#include <shared_mutex>

#include "Geometry/Mesh.hpp"
#include "Particle/Particle.hpp"
#include "ParticleInCellEngine/PICTypes.hpp"

/**
 * @class SurfaceCollisionHandler
 * @brief Handles collisions between particles and surface meshes.
 *
 * This class detects and resolves collisions of particles with the surface mesh,
 * updating settlement states and recording intersection points.
 */
class SurfaceCollisionHandler
{
private:
    std::shared_mutex &m_settledParticlesMutex;               ///< Mutex for protecting settled particle data.
    std::mutex &m_particlesMovementMutex;                     ///< Mutex for protecting particle movement data.
    AABB_Tree_Triangle const &m_surfaceMeshAABBtree;          ///< AABB tree for efficient collision detection.
    MeshTriangleParamVector const &m_triangleMesh;            ///< Vector of mesh triangle parameters.
    ParticlesIDSet &m_settledParticleIds;                     ///< Set of settled particle IDs.
    SettledParticlesCounterMap &m_settledParticlesCounterMap; ///< Map of triangle IDs to settled particle counts.
    ParticleMovementMap &m_particlesMovement;                 ///< Map of particle movements.

public:
    /**
     * @brief Constructs a SurfaceCollisionHandler object.
     * @param settledParticlesMutex Mutex for settled particle data.
     * @param particlesMovementMutex Mutex for particle movement data.
     * @param tree AABB tree for surface mesh collision detection.
     * @param mesh Vector of mesh triangle parameters.
     * @param settledParticlesIds Reference to the set of settled particle IDs.
     * @param settledParticlesCounterMap Reference to the map of triangle settlement counts.
     * @param particlesMovement Reference to the map of particle movements.
     */
    SurfaceCollisionHandler(
        std::shared_mutex &settledParticlesMutex,
        std::mutex &particlesMovementMutex,
        AABB_Tree_Triangle const &tree,
        MeshTriangleParamVector const &mesh,
        ParticlesIDSet &settledParticlesIds,
        SettledParticlesCounterMap &settledParticlesCounterMap,
        ParticleMovementMap &particlesMovement);

    /**
     * @brief Handles a collision between a particle and the surface mesh.
     * @param particle The particle involved in the collision.
     * @param ray The ray representing the particle's movement.
     * @param particlesNumber Total number of particles in the simulation.
     * @return The ID of the triangle where the particle collided, or `std::nullopt` if no collision occurred.
     */
    std::optional<size_t> handle(Particle const &particle, Ray const &ray, size_t particlesNumber);
};

#endif // SURFACECOLLISIONHANDLER_HPP
