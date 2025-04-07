#ifndef PARTICLE_SURFACE_COLLISION_HANDLER_HPP
#define PARTICLE_SURFACE_COLLISION_HANDLER_HPP

#include <shared_mutex>

#include "Geometry/Mesh/Surface/SurfaceMesh.hpp"
#include "ParticleInCellEngine/PICTypes.hpp"
#include "ParticleInCellEngine/ParticleDynamicsProcessor/StopModelingObserver.hpp"

/**
 * @class ParticleSurfaceCollisionHandler
 * @brief Handles collisions of particles with a surface mesh in a particle-in-cell (PIC) simulation.
 *
 * The `ParticleSurfaceCollisionHandler` class is responsible for detecting and processing interactions
 * between a moving particle and a surface mesh represented by a triangle-based structure. When a particle's
 * trajectory intersects with a triangle in the mesh, the handler updates the number of settled particles
 * on that surface and records the particle's movement. Additionally, the handler is responsible for stopping
 * the simulation if the required number of particles has settled.
 *
 * @details The class operates in a multi-threaded environment, utilizing `std::shared_mutex` and `std::mutex`
 * to ensure thread-safe updates of particle data structures. The collision detection is performed using an
 * Axis-Aligned Bounding Box (AABB) tree to accelerate intersection queries between particle trajectories and
 * the surface mesh.
 *
 * @note This class provides only a static method and is not intended to be instantiated.
 *
 * **Key Responsibilities:**
 * - Detects collisions between particles and the surface mesh using an AABB tree.
 * - Identifies the triangle ID with which the particle collides.
 * - Updates the number of settled particles for the intersected triangle.
 * - Stops the simulation when the required number of particles has been settled.
 * - Records the movement of the particle after collision.
 *
 * @warning This class assumes that the `SurfaceMesh` has been properly initialized before calling the `handle` method.
 *
 * @thread_safety
 * - Uses `std::shared_mutex` for safe read/write access to the settled particle count map.
 * - Uses `std::mutex` to ensure safe updates to the particle movement map.
 *
 * **Usage Example:**
 * @code
 * Particle particle = Particle::generateRandom();
 * Ray particleRay = Ray(particle.getPosition(), particle.getVelocity());
 * SurfaceMesh surface("mesh.msh");
 * std::shared_mutex settledMutex;
 * std::mutex movementMutex;
 * ParticleMovementMap movementMap;
 * StopSubject stopSignal;
 *
 * ParticleSurfaceCollisionHandler::handle(particle, particleRay, 1000, surface,
 *                                         settledMutex, movementMutex, movementMap, stopSignal);
 * @endcode
 */
class ParticleSurfaceCollisionHandler
{
public:
    /**
     * @brief Handles the collision of a single particle with a surface mesh.
     *
     * This method determines if a particle collides with the surface mesh and, if so,
     * updates the count of settled particles, records the movement, and stops the simulation
     * if the required number of particles have settled.
     *
     * @param[in] particle The particle whose trajectory is being checked for intersection.
     * @param[in] ray The ray representing the particle's trajectory.
     * @param[in] totalParticles The total number of particles required to stop the simulation.
     * @param[in,out] surfaceMesh The surface mesh with which the particle may collide.
     * @param[in,out] sh_mutex_settledParticlesCounterMap Shared mutex for synchronizing settled particle updates.
     * @param[in,out] mutex_particlesMovementMapMutex Mutex for synchronizing updates to the particle movement map.
     * @param[in,out] particleMovementMap A map storing the trajectory of particles post-collision.
     * @param[in,out] settledParticlesIds Set of the particle IDs that are been settled (need to avoid checking already settled particles).
     * @param[in,out] stopSubject Object used to notify the system when the required number of particles have settled.
     * @return Triangle ID with which particle intersected. if there is no any intersection or no corresponding triangle - returning std::nullopt.
     *
     * @details
     * - Uses an **AABB tree** to quickly determine if the particle collides with the surface.
     * - If a collision is detected, checks if the intersected triangle exists in the `SurfaceMesh`.
     * - Increments the count of settled particles for the intersected triangle.
     * - If the total number of settled particles reaches `particlesNumber`, stops the simulation.
     * - Records the intersection point in `particleMovementMap` for further analysis.
     *
     * @return void
     *
     * @exception noexcept
     * This function does not throw exceptions.
     *
     * @thread_safety
     * - **Thread-safe** due to the use of `std::shared_mutex` for settled particles count.
     * - **Thread-safe** due to the use of `std::mutex` for updating particle movements.
     *
     * **Algorithm:**
     * 0. Check if particle is already settled - skip further processing.
     * 1. Check if the ray intersects with any triangle in the **AABB tree** of the surface mesh.
     * 2. Retrieve the intersected triangle and verify if it is a valid triangle in `SurfaceMesh`.
     * 3. If valid, find the triangle ID from `SurfaceMesh::getTriangleCellMap()`.
     * 4. Lock the **settled particles counter** and increment the count.
     * 5. If the total settled particles exceed `particlesNumber`, stop the simulation.
     * 6. Lock the **particle movement map** and store the intersection point.
     * 7. Adding ID of the particle to avoid processing already settled particle.
     *
     * **Example Usage:**
     * @code
     * Particle particle = Particle::generateRandom();
     * Ray particleRay = Ray(particle.getPosition(), particle.getVelocity());
     * SurfaceMesh surface("mesh.msh");
     * std::shared_mutex settledMutex;
     * std::mutex movementMutex;
     * ParticleMovementMap movementMap;
     * StopSubject stopSignal;
     *
     * ParticleSurfaceCollisionHandler::handle(particle, particleRay, 1000, surface,
     *                                         settledMutex, movementMutex, movementMap, stopSignal);
     * @endcode
     *
     * @warning Ensure that the `SurfaceMesh` is properly initialized before invoking this method.
     */
    static std::optional<size_t> handle(Particle_cref particle,
                                        Segment_cref segment,
                                        size_t totalParticles,
                                        SurfaceMesh_ref surfaceMesh,
                                        std::shared_mutex &sh_mutex_settledParticlesCounterMap,
                                        std::mutex &mutex_particlesMovementMapMutex,
                                        ParticleMovementMap_ref particleMovementMap,
                                        ParticlesIDSet_ref settledParticlesIds,
                                        StopSubject &stopSubject);
};

#endif // !PARTICLE_SURFACE_COLLISION_HANDLER_HPP
