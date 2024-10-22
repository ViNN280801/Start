#ifndef PARTICLEGENERATOR_HPP
#define PARTICLEGENERATOR_HPP

#if __cplusplus < 202002L
#include <type_traits>
#include <utility> // for std::declval
#endif

#include "Particle/Particle.hpp"

#if __cplusplus >= 202002L
/**
 * @brief Concept to ensure the Generator is callable and returns a Particle.
 *
 * This concept requires that the Generator type be a callable (such as a function, lambda,
 * or functor) that returns a `Particle` or a type convertible to `Particle`.
 */
template <typename Generator>
concept ParticleGeneratorConcept = std::invocable<Generator> && std::same_as<std::invoke_result_t<Generator>, Particle>;

#else

template <typename Generator>
constexpr bool ParticleGeneratorConcept_v = std::is_invocable_r_v<Particle, Generator>;

// SFINAE check for C++17
template <typename Generator>
using ParticleGeneratorConcept = std::enable_if_t<ParticleGeneratorConcept_v<Generator>, bool>;

#endif

/**
 * @brief The ParticleGenerator class provides various methods to generate particles with different properties.
 *
 * This class offers multiple static methods to create particles with specific initial positions and velocities.
 * Particles can be generated with randomized or fixed properties, such as within specified spatial and velocity
 * ranges, based on velocity magnitude, or using pre-defined particle sources. The class supports both point
 * and surface sources for particle generation.
 *
 * The core particle generation logic is encapsulated in a private template function, `_generate`, which accepts
 * a callable generator function that defines how individual particles are constructed. This approach provides
 * flexibility and allows for efficient particle creation using lambda functions.
 *
 * @note This class uses the Resource Acquisition Is Initialization (RAII) approach and leverages OpenMP
 *       (if enabled) for parallel particle generation. The user must ensure that the Particle type
 *       and any custom generator functions are compatible with this approach.
 *
 * Example Usage:
 * @code
 * // Generating 100 particles with randomized positions and velocities.
 * ParticleVector particles = ParticleGenerator::byVelocities(
 *     100, ParticleType::Ar,
 *     0.0, 0.0, 0.0, 100.0, 100.0, 100.0,
 *     -50.0, -50.0, -50.0, 50.0, 50.0, 50.0
 * );
 *
 * // Generating particles from a point source.
 * std::vector<point_source_t> pointSources = {...};
 * ParticleVector pointParticles = ParticleGenerator::fromPointSource(pointSources);
 * @endcode
 */
class ParticleGenerator
{
private:
/**
 * @brief Creates a specified number of particles using a generator function.
 *
 * This function generates particles by invoking a provided generator function. The generator
 * must return a `Particle` or an object convertible to `Particle`.
 *
 * @tparam Gen Type of the generator function, constrained by the `ParticleGeneratorConcept` concept.
 * @param count The number of particles to generate.
 * @param gen A callable (e.g., lambda, function) that generates a `Particle` when invoked.
 * @return ParticleVector A vector containing the generated particles.
 *
 * @note The generator must return a `Particle` object on each invocation.
 * @code
 * ParticleVector particles = createParticles(100, []() {
 *     return Particle(...);  // Return a particle instance
 * });
 * @endcode
 */
#if __cplusplus >= 202002L
        template <ParticleGeneratorConcept Gen>
#else
        template <typename Gen, typename = ParticleGeneratorConcept<Gen>>
#endif
        static ParticleVector _generate(size_t count, Gen gen)
        {
                ParticleVector particles;

#ifdef USE_OMP
                particles.reserve(count);
#pragma omp parallel for simd
                for (size_t i = 0; i < count; ++i)
                {
                        particles[i] = gen();
                }
#else
                for (size_t i{}; i < count; ++i)
                        particles.emplace_back(gen());
#endif

                return particles;
        }

public:
        /**
         * @brief Generates particles with random positions and velocities within specified ranges.
         * @details This function uses a lambda to create particles with randomized positions and velocities,
         *          utilizing a real number generator to provide values within the given min and max ranges for
         *          each component.
         *
         * @param count The number of particles to generate.
         * @param type The type of the particles to generate.
         * @param minx Minimum x-coordinate for particle position.
         * @param miny Minimum y-coordinate for particle position.
         * @param minz Minimum z-coordinate for particle position.
         * @param maxx Maximum x-coordinate for particle position.
         * @param maxy Maximum y-coordinate for particle position.
         * @param maxz Maximum z-coordinate for particle position.
         * @param minvx Minimum x-component of particle velocity.
         * @param minvy Minimum y-component of particle velocity.
         * @param minvz Minimum z-component of particle velocity.
         * @param maxvx Maximum x-component of particle velocity.
         * @param maxvy Maximum y-component of particle velocity.
         * @param maxvz Maximum z-component of particle velocity.
         * @return ParticleVector containing the generated particles.
         */
        static ParticleVector byVelocities(size_t count, ParticleType type,
                                           double minx, double miny, double minz,
                                           double maxx, double maxy, double maxz,
                                           double minvx, double minvy, double minvz,
                                           double maxvx, double maxvy, double maxvz);

        /**
         * @brief Generates particles with specified fixed positions and velocities.
         * @details This function creates particles by calling a lambda that generates each particle
         *          with fixed position and velocity values.
         *
         * @param count The number of particles to generate.
         * @param type The type of the particles to generate.
         * @param x Fixed x-coordinate for particle position.
         * @param y Fixed y-coordinate for particle position.
         * @param z Fixed z-coordinate for particle position.
         * @param vx Fixed x-component of particle velocity.
         * @param vy Fixed y-component of particle velocity.
         * @param vz Fixed z-component of particle velocity.
         * @return ParticleVector containing the generated particles.
         */
        static ParticleVector byVelocities(size_t count, ParticleType type,
                                           double x, double y, double z,
                                           double vx, double vy, double vz);

        /**
         * @brief Generates particles with specified positions and velocities based on a velocity magnitude.
         * @details The method uses spherical coordinates to generate velocity components (vx, vy, vz) from the
         *          given magnitude `v`, angle `theta`, and azimuthal angle `phi`. These angles are randomized
         *          within their given ranges using a real number generator.
         *
         * @param count The number of particles to generate.
         * @param type The type of the particles to generate.
         * @param x Fixed x-coordinate for particle position.
         * @param y Fixed y-coordinate for particle position.
         * @param z Fixed z-coordinate for particle position.
         * @param v Magnitude of the velocity.
         * @param theta Range for the polar angle (0 to theta).
         * @param phi Range for the azimuthal angle (0 to phi).
         * @return ParticleVector containing the generated particles.
         */
        static ParticleVector byVelocityModule(size_t count, ParticleType type,
                                               double x, double y, double z,
                                               double v, double theta, double phi);

        /**
         * @brief Creates a vector of particles using particle sources as points.
         * @param source A vector of point particle sources.
         * @return A vector of particles created from the given point particle sources.
         * @details This function iterates through the provided point particle sources,
         *          and for each source, it generates the specified number of particles.
         *          Each particle is assigned its type, position, energy, and direction
         *          angles (theta, phi, expansionAngle) based on the source parameters.
         *
         *          1) The function goes through each source.
         *          2) Creates a set number of particles for each source, setting the type, position, energy and directions
         *             for each (angles theta, phi, expansionAngle).
         */
        static ParticleVector fromPointSource(std::vector<point_source_t> const &source);

        /**
         * @brief Creates a vector of particles using particle sources as surfaces.
         * @param source A vector of surface particle sources.
         * @return A vector of particles created from the given surface particle sources.
         * @details This function iterates through the provided surface particle sources,
         *          and for each source, it distributes particles evenly across the
         *          specified cell centers. If the number of particles does not divide
         *          evenly among the cells, the remainder is randomly distributed. Each
         *          particle is assigned its type, position, energy, and direction based
         *          on the source parameters and cell normals.
         *
         *          1) The function passes through each surface source.
         *          2) Determines the number of cells and the number of particles per cell.
         *          3) Distributes the remainder of the particles randomly into cells.
         *          4) For each cell and normal, calculates the angles theta and phi necessary to determine the direction of the particles.
         *          5) Creates particles by setting for each type, position, energy and directions (angles theta, phi, expansionAngle).
         */
        static ParticleVector fromSurfaceSource(std::vector<surface_source_t> const &source);
};

#endif // !PARTICLEGENERATOR_HPP