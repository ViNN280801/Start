#ifndef PARTICLE_DISPLACER_HPP
#define PARTICLE_DISPLACER_HPP

#include <boost/functional/hash.hpp>
#include <unordered_map>

#include "Particle/Particle.hpp"

/**
 * @brief Class for efficiently calculating and applying particle displacements from sources.
 * @details Calculates optimized displacement vectors for particles based on their source directions
 *          and caches the results to avoid redundant calculations.
 */
class ParticleDisplacer final
{
private:
    /// @brief Structure to hold direction vector components
    struct DirectionVector
    {
        double x;
        double y;
        double z;

        bool operator==(DirectionVector const &other) const;
    };

    /// @brief Hash function for DirectionVector to use in unordered_map
    struct DirectionVectorHash
    {
        std::size_t operator()(DirectionVector const &direction) const;
    };

    static std::unordered_map<DirectionVector, CGALVector, DirectionVectorHash> s_displacementCache; ///< Cache to store calculated displacement vectors
    static constexpr double DISPLACEMENT_FACTOR = 0.05;                                              ///< Displacement magnitude as a percentage of direction vector (5% default)

    /**
     * @brief Calculates the displacement vector for a given direction.
     * @details Implements the algorithm:
     *          1) Find the maximum component of the direction vector
     *          2) Normalize the vector with respect to this maximum
     *          3) Scale it by the displacement factor
     * @param dirX X component of the direction
     * @param dirY Y component of the direction
     * @param dirZ Z component of the direction
     * @return Calculated displacement vector
     */
    static CGALVector _calculateDisplacement(double dirX, double dirY, double dirZ);

    /**
     * @brief Gets a displacement vector from the cache or calculates a new one.
     * @param direction The direction vector
     * @return The corresponding displacement vector
     */
    static CGALVector _getDisplacementVector(DirectionVector const &direction);

public:
    /**
     * @brief Displaces particles from point sources based on their configured directions.
     * @param particles Vector of particles to be displaced
     * @param pointSources Vector of point sources containing direction information
     */
    static void displaceParticlesFromPointSources(ParticleVector &particles,
                                                  std::vector<point_source_t> const &pointSources);

    /**
     * @brief Displaces particles from surface sources based on surface normals.
     * @param particles Vector of particles to be displaced
     * @param surfaceSources Vector of surface sources containing normal information
     */
    static void displaceParticlesFromSurfaceSources(ParticleVector &particles,
                                                    std::vector<surface_source_t> const &surfaceSources);

    /**
     * @brief Displaces particles based on their source type (point or surface).
     * @param particles Vector of particles to be displaced
     * @param pointSources Vector of point sources
     * @param surfaceSources Vector of surface sources
     */
    static void displaceParticles(ParticleVector &particles,
                                  std::vector<point_source_t> const &pointSources,
                                  std::vector<surface_source_t> const &surfaceSources);
};

#endif // !PARTICLE_DISPLACER_HPP
