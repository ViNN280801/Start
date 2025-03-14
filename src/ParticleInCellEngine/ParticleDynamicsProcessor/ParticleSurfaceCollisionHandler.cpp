#include "ParticleInCellEngine/ParticleDynamicsProcessor/ParticleSurfaceCollisionHandler.hpp"
#include "Geometry/Utils/Intersections/SegmentTriangleIntersection.hpp"
#include "ParticleInCellEngine/ParticleDynamicsProcessor/ParticleMovementTracker.hpp"
#include "ParticleInCellEngine/ParticleDynamicsProcessor/ParticleSettler.hpp"

std::optional<size_t> ParticleSurfaceCollisionHandler::handle(Particle_cref particle,
                                                              Segment_cref segment,
                                                              size_t totalParticles,
                                                              SurfaceMesh_ref surfaceMesh,
                                                              std::shared_mutex &sh_mutex_settledParticlesCounterMap,
                                                              std::mutex &mutex_particlesMovementMapMutex,
                                                              ParticleMovementMap_ref particleMovementMap,
                                                              ParticlesIDSet_ref settledParticlesIds,
                                                              StopSubject &stopSubject)
{
    // 0. If particle is already settled - skip it.
    if (ParticleSettler::isSettled(particle.getId(), settledParticlesIds, sh_mutex_settledParticlesCounterMap))
        return std::nullopt;

    // 1. If there any intersection of the ray and current AABB tree of the surface mesh.
    auto intersection{surfaceMesh.getAABBTree().any_intersection(segment)};
    if (!intersection)
        return std::nullopt;

    // 2. Getting triangle with which ray intersected.
    auto triangle{*intersection->second};
    if (triangle.is_degenerate())
        return std::nullopt;

// 3. If this triangle is really in surface mesh, getting triangle iterator to get it Id.
#if __cplusplus >= 202002L
    auto triangleIdIter{std::ranges::find_if(surfaceMesh.getTriangleCellMap(), [triangle](auto const &entry)
                                             {
        auto const &[id, cellData]{entry};
        return cellData.triangle == triangle; })};
#else
    auto triangleIdIter{std::find_if(surfaceMesh.getTriangleCellMap().cbegin(), surfaceMesh.getTriangleCellMap().cend(), [triangle](auto const &entry)
                                     {
        auto const &[id, cellData]{entry};
        return cellData.triangle == triangle; })};
#endif

    // 4. If there is no such triangle in mesh, just skip all further processing.
    if (triangleIdIter == surfaceMesh.getTriangleCellMap().cend())
        return std::nullopt;

    // 5. Getting triangle ID as a 'size_t'.
    size_t triangleId{triangleIdIter->first};

    // 6. Settling particle.
    ParticleSettler::settle(particle.getId(),
                            triangleId,
                            totalParticles,
                            surfaceMesh,
                            sh_mutex_settledParticlesCounterMap,
                            settledParticlesIds,
                            stopSubject);

    // 7. Recording particle movement.
    // 7.1. Calculating intersection point.
    auto intersectionPoint{SegmentTriangleIntersection::getIntersectionPoint(segment, triangle)};

    // 7.2. If optional is not empty - add it to the particle movement map.
    if (intersectionPoint.has_value())
    {
        ParticleMovementTracker::recordMovement(particleMovementMap,
                                                mutex_particlesMovementMapMutex,
                                                particle.getId(),
                                                intersectionPoint.value());
    }

    // 8. Returning triangle ID with which ray intersected.
    return triangleId;
}
