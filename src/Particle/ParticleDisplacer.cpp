#include "Particle/ParticleDisplacer.hpp"

std::unordered_map<ParticleDisplacer::DirectionVector, CGALVector, ParticleDisplacer::DirectionVectorHash>
    ParticleDisplacer::s_displacementCache;

bool ParticleDisplacer::DirectionVector::operator==(DirectionVector const &other) const
{
    return x == other.x && y == other.y && z == other.z;
}

std::size_t ParticleDisplacer::DirectionVectorHash::operator()(DirectionVector const &direction) const
{
    std::size_t seed{};
    boost::hash_combine(seed, boost::hash_value(direction.x));
    boost::hash_combine(seed, boost::hash_value(direction.y));
    boost::hash_combine(seed, boost::hash_value(direction.z));
    return seed;
}

CGALVector ParticleDisplacer::_calculateDisplacement(double dirX, double dirY, double dirZ)
{
    // Find maximum absolute component
    double maxComponent{std::max({std::fabs(dirX), std::fabs(dirY), std::fabs(dirZ)})};

    // Avoid division by zero
    if (maxComponent < std::numeric_limits<double>::epsilon())
        return CGALVector(0.0, 0.0, 0.0);

    // Normalize vector with respect to max component and apply displacement factor
    double normX{(dirX / maxComponent) * DISPLACEMENT_FACTOR},
        normY{(dirY / maxComponent) * DISPLACEMENT_FACTOR},
        normZ{(dirZ / maxComponent) * DISPLACEMENT_FACTOR};

    return CGALVector(normX, normY, normZ);
}

CGALVector ParticleDisplacer::_getDisplacementVector(DirectionVector const &direction)
{
    auto it{s_displacementCache.find(direction)};
    if (it != s_displacementCache.end())
        return it->second;

    CGALVector displacement{_calculateDisplacement(direction.x, direction.y, direction.z)};
    s_displacementCache[direction] = displacement;
    return displacement;
}

void ParticleDisplacer::displaceParticlesFromPointSources(ParticleVector &particles,
                                                          std::vector<point_source_t> const &pointSources)
{
    if (particles.empty())
    {
        WARNINGMSG("Warning: There are no particles to displace, skipping displacement");
        return;
    }
    if (pointSources.empty())
    {
        WARNINGMSG("Warning: There are no point sources to displace particles, skipping displacement");
        return;
    }

    // Process all point sources and create directions map
    std::unordered_map<Point, DirectionVector, boost::hash<Point>> sourceDirections;
    for (auto const &source : pointSources)
    {
        Point basePoint(source.baseCoordinates[0], source.baseCoordinates[1], source.baseCoordinates[2]);

        // Convert spherical angles (phi, theta) to Cartesian direction vector
        double sinTheta{std::sin(source.theta)},
            dirX{sinTheta * std::cos(source.phi)},
            dirY{sinTheta * std::sin(source.phi)},
            dirZ{std::cos(source.theta)};

        sourceDirections[basePoint] = {dirX, dirY, dirZ};
    }

    // Apply displacement to particles
    for (auto &particle : particles)
    {
        Point const &center{particle.getCentre()};
        if (auto it{sourceDirections.find(center)}; it != sourceDirections.end())
        {
            CGALVector displacement{_getDisplacementVector(it->second)};
            particle.setCentre(center + displacement);
        }
    }
}

void ParticleDisplacer::displaceParticlesFromSurfaceSources(ParticleVector &particles,
                                                            std::vector<surface_source_t> const &surfaceSources)
{
    if (particles.empty())
    {
        WARNINGMSG("Warning: There are no particles to displace, skipping displacement");
        return;
    }
    if (surfaceSources.empty())
    {
        WARNINGMSG("Warning: There are no surface sources to displace particles, skipping displacement");
        return;
    }

    // Create a map of base coordinates to direction vectors for each surface point
    std::unordered_map<Point, DirectionVector, boost::hash<Point>> sourceDirections;
    for (auto const &source : surfaceSources)
    {
        auto const &baseCoords{source.baseCoordinates};
        for (auto const &[coordStr, normalVector] : baseCoords)
        {
            // Parse the coordinate string (format: "x, y, z")
            std::istringstream ss(coordStr);
            double x, y, z;
            char comma;

            if (ss >> x >> comma >> y >> comma >> z)
            {
                Point point(x, y, z);
                if (normalVector.size() >= 3)
                    sourceDirections[point] = {normalVector[0], normalVector[1], normalVector[2]};
            }
        }
    }

    // Apply displacement to particles
    for (auto &particle : particles)
    {
        Point const &center{particle.getCentre()};
        if (auto it{sourceDirections.find(center)}; it != sourceDirections.end())
        {
            CGALVector displacement{_getDisplacementVector(it->second)};
            particle.setCentre(center + displacement);
        }
    }
}

void ParticleDisplacer::displaceParticles(ParticleVector &particles,
                                          std::vector<point_source_t> const &pointSources,
                                          std::vector<surface_source_t> const &surfaceSources)
{
    displaceParticlesFromPointSources(particles, pointSources);
    displaceParticlesFromSurfaceSources(particles, surfaceSources);
}