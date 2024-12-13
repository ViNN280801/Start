#include "ParticleInCellEngine/ParticleDynamicsProcessor/ParticleSurfaceCollisionHandler.hpp"

ParticleSurfaceCollisionHandler::ParticleSurfaceCollisionHandler(
    std::shared_mutex &settledParticlesMutex,
    std::mutex &particlesMovementMutex,
    AABB_Tree_Triangle const &tree,
    MeshTriangleParamVector const &mesh,
    ParticlesIDSet &settledParticlesIds,
    SettledParticlesCounterMap &settledParticlesCounterMap,
    ParticleMovementMap &particlesMovement,
    StopSubject &subject) : m_settledParticlesMutex(settledParticlesMutex),
                            m_particlesMovementMutex(particlesMovementMutex),
                            m_surfaceMeshAABBtree(tree),
                            m_triangleMesh(mesh),
                            m_settledParticleIds(settledParticlesIds),
                            m_settledParticlesCounterMap(settledParticlesCounterMap),
                            m_particlesMovement(particlesMovement),
                            m_subject(subject) {}

std::optional<size_t> ParticleSurfaceCollisionHandler::handle(Particle const &particle, Ray const &ray, size_t particlesNumber)
{
    auto intersection{m_surfaceMeshAABBtree.any_intersection(ray)};
    if (!intersection)
        return std::nullopt;

    auto triangle{*intersection->second};
    if (triangle.is_degenerate())
        return std::nullopt;

#if __cplusplus >= 202002L
    auto matchedIt{std::ranges::find_if(m_triangleMesh, [triangle](auto const &el)
                                        { return triangle == std::get<1>(el); })};
#else
    auto matchedIt{std::find_if(m_triangleMesh.cbegin(), m_triangleMesh.cend(), [triangle](auto const &el)
                                { return triangle == std::get<1>(el); })};
#endif

    if (matchedIt != m_triangleMesh.end())
    {
        auto id{Mesh::isRayIntersectTriangle(ray, *matchedIt)};
        if (id)
        {
            {
                std::unique_lock<std::shared_mutex> lock(m_settledParticlesMutex);
                ++m_settledParticlesCounterMap[id.value()];
                m_settledParticleIds.insert(particle.getId());

                if (m_settledParticleIds.size() >= particlesNumber)
                {
                    m_subject.notifyStopRequested();
                    return std::nullopt;
                }
            }

            {
                std::lock_guard<std::mutex> lock(m_particlesMovementMutex);
                auto intersection_point{RayTriangleIntersection::getIntersectionPoint(ray, triangle)};
                if (intersection_point)
                    m_particlesMovement[particle.getId()].emplace_back(*intersection_point);
            }
        }
    }

    return std::nullopt;
}
