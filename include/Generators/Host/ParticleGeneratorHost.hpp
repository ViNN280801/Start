#ifndef PARTICLEGENERATORHOST_HPP
#define PARTICLEGENERATORHOST_HPP

#if __cplusplus < 202002L
#include <type_traits>
#include <utility> // for std::declval
#endif

#include "Particle/Particle.hpp"
#include "Utilities/ConfigParser.hpp"

#if __cplusplus >= 202002L
/**
 * @brief Concept to ensure the Generator is callable and returns a Particle.
 *
 * This concept requires that the Generator type be a callable (such as a function, lambda,
 * or functor) that returns a `Particle` or a type convertible to `Particle`.
 */
template <typename Generator>
concept ParticleGeneratorHostConcept = std::invocable<Generator> && std::same_as<std::invoke_result_t<Generator>, Particle>;

#else

template <typename Generator>
constexpr bool ParticleGeneratorHostConcept_v = std::is_invocable_r_v<Particle, Generator>;

// SFINAE check for C++17
template <typename Generator>
using ParticleGeneratorHostConcept = std::enable_if_t<ParticleGeneratorHostConcept_v<Generator>, bool>;
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
class ParticleGeneratorHost
{
public:
/**
 * @brief Creates a specified number of particles using a generator function.
 *
 * This function generates particles by invoking a provided generator function. The generator
 * must return a `Particle` or an object convertible to `Particle`.
 *
 * @tparam Gen Type of the generator function, constrained by the `ParticleGeneratorHostConcept` concept.
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
        template <ParticleGeneratorHostConcept Gen>
#else
        template <typename Gen, typename = ParticleGeneratorHostConcept<Gen>>
#endif
        static ParticleVector generate(size_t count, Gen gen)
        {
                if (count == 0)
                        throw std::logic_error("There is no need to generate 0 objects");

                ParticleVector particles(count);

#ifdef USE_OMP
                // Selecting chunk size to correctly handle different cases.
                size_t chunkSize{1000};
                if (count <= 5000)
                        chunkSize = 500;
                else if (count <= 1000)
                        chunkSize = 100;
                else if (count <= 100)
                        chunkSize = 10;
                else
                        chunkSize = 1;

#pragma omp parallel for simd schedule(static, chunkSize)
#endif
                for (size_t i = 0; i < count; ++i)
                        particles[i] = gen();
                return particles;
        }

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

#endif // !PARTICLEGENERATORHOST_HPP
