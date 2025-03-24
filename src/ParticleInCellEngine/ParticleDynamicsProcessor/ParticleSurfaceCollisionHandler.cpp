#include "ParticleInCellEngine/ParticleDynamicsProcessor/ParticleSurfaceCollisionHandler.hpp"
#include "Geometry/Utils/Intersections/SegmentTriangleIntersection.hpp"
#include "ParticleInCellEngine/ParticleDynamicsProcessor/ParticleMovementTracker.hpp"
#include "ParticleInCellEngine/ParticleDynamicsProcessor/ParticleSettler.hpp"
#include <vector>

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

    // 2. Find all intersections to determine the closest one
    std::vector<std::pair<Point, size_t>> intersections;

    // Using a temporary vector to collect all AABB intersections
    std::vector<typename AABBTree::Primitive::Id> primitives;
    surfaceMesh.getAABBTree().all_intersected_primitives(segment, std::back_inserter(primitives));

    for (const auto &primId : primitives)
    {
        // Get the triangle from the primitive ID
        auto triangle{*primId};
        if (triangle.is_degenerate())
            continue;

        // Find the triangle in the mesh to get its ID
        auto triangleIdIter{std::find_if(
            surfaceMesh.getTriangleCellMap().cbegin(),
            surfaceMesh.getTriangleCellMap().cend(),
            [&triangle](auto const &entry)
            {
                auto const &[id, cellData] = entry;
                return cellData.triangle == triangle;
            })};

        if (triangleIdIter == surfaceMesh.getTriangleCellMap().cend())
            continue;

        size_t triangleId{triangleIdIter->first};

        // Calculate the actual intersection point
        auto intersectionPoint{SegmentTriangleIntersection::getIntersectionPoint(segment, triangle)};
        if (!intersectionPoint.has_value())
            continue;

        // Store both the intersection point and triangle ID
        intersections.emplace_back(*intersectionPoint, triangleId);
    }

    // If no valid intersections were found (unlikely but possible)
    if (intersections.empty())
    {
        // Fallback to the original behavior using any_intersection
        auto triangle{*intersection->second};
        if (triangle.is_degenerate())
            return std::nullopt;

        auto triangleIdIter{std::find_if(surfaceMesh.getTriangleCellMap().cbegin(),
                                         surfaceMesh.getTriangleCellMap().cend(),
                                         [triangle](auto const &entry)
                                         {
                                             auto const &[id, cellData]{entry};
                                             return cellData.triangle == triangle;
                                         })};

        if (triangleIdIter == surfaceMesh.getTriangleCellMap().cend())
            return std::nullopt;

        size_t triangleId{triangleIdIter->first};
        ParticleSettler::settle(particle.getId(),
                                triangleId,
                                totalParticles,
                                surfaceMesh,
                                sh_mutex_settledParticlesCounterMap,
                                settledParticlesIds,
                                stopSubject);

        // Calculate intersection point
        auto intersectionPoint{SegmentTriangleIntersection::getIntersectionPoint(segment, triangle)};
        if (intersectionPoint.has_value())
        {
            ParticleMovementTracker::recordMovement(particleMovementMap,
                                                    mutex_particlesMovementMapMutex,
                                                    particle.getId(),
                                                    intersectionPoint.value());
        }

        return triangleId;
    }

    // 3. Find the closest intersection to the starting point
    Point startPoint{segment.source()};
    double minDistance{std::numeric_limits<double>::max()};
    size_t closestTriangleId{};
    Point closestIntersection;

    for (const auto &[point, triangleId] : intersections)
    {
        double dist{std::sqrt(
            std::pow(point.x() - startPoint.x(), 2) +
            std::pow(point.y() - startPoint.y(), 2) +
            std::pow(point.z() - startPoint.z(), 2))};

        if (dist < minDistance)
        {
            minDistance = dist;
            closestTriangleId = triangleId;
            closestIntersection = point;
        }
    }

    // 4. Settling particle on the closest intersection
    ParticleSettler::settle(particle.getId(),
                            closestTriangleId,
                            totalParticles,
                            surfaceMesh,
                            sh_mutex_settledParticlesCounterMap,
                            settledParticlesIds,
                            stopSubject);

    // 5. Record the particle movement at the intersection point
    ParticleMovementTracker::recordMovement(particleMovementMap,
                                            mutex_particlesMovementMapMutex,
                                            particle.getId(),
                                            closestIntersection);

    // 6. Return the triangle ID of the intersection
    return closestTriangleId;
}
