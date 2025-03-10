#ifndef PARTICLEGENERATORDEVICE_CUH
#define PARTICLEGENERATORDEVICE_CUH

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "Particle/CUDA/ParticleDevice.cuh"
#include "Particle/Particle.hpp"
#include "Utilities/ConfigParser.hpp"

/**
 * @class ParticleGeneratorDevice
 * @brief A class for generating particles on the GPU from point and surface sources.
 *
 * The `ParticleGeneratorDevice` class provides methods to create and initialize particles
 * based on given source descriptions. Sources can be either point sources or surface sources.
 * This class utilizes CUDA for efficient particle generation and initialization on the device.
 *
 * ### Key Features
 * - Supports point and surface particle sources.
 * - Uses GPU kernels for fast particle initialization.
 * - Manages particle attributes such as position, energy, velocity, and direction.
 *
 * ### Usage
 * - Call `fromPointSource` to generate particles from point sources.
 * - Call `fromSurfaceSource` to generate particles from surface sources.
 */
class ParticleGeneratorDevice
{
public:
    /**
     * @brief Generates particles from point sources.
     * @param source A vector of point particle sources.
     * @return A `ParticleVector` containing the generated particles.
     * @details Each point source specifies a position, particle type, energy, and direction angles.
     *          The function uses a GPU kernel to generate particles for each source, assigning
     *          properties such as type, position, energy, and velocity. The velocity is computed
     *          based on energy and direction angles (theta, phi, expansionAngle).
     *
     * Steps:
     * 1. Iterate through the provided point sources.
     * 2. Launch a GPU kernel for each source to generate the specified number of particles.
     * 3. Assign attributes such as type, position, energy, and direction for each particle.
     */
    static ParticleVector fromPointSource(std::vector<point_source_t> const &source);

    /**
     * @brief Generates particles from surface sources.
     * @param source A vector of surface particle sources.
     * @param expansionAngle Expansion angle in [rad] for the cone distribution (by default = 0). Assuming that there is no expansion in surface source.
     * @return A `ParticleVector` containing the generated particles.
     * @details Surface sources define particle distribution across cell centers. Each surface source
     *          specifies the cell geometry, particle type, and energy. Particles are distributed evenly
     *          across the cells, with any remainder randomly assigned. Velocity is computed based on
     *          the cell normals and particle energy.
     *
     * Steps:
     * 1. Distribute particles evenly across the cells in each surface source.
     * 2. Assign attributes such as type, position, energy, and velocity for each particle.
     * 3. Compute particle velocity based on the cell normal vectors.
     */
    static ParticleVector fromSurfaceSource(std::vector<surface_source_t> const &source, double expansionAngle = 0.0);
};

#endif // !USE_CUDA

#endif // !PARTICLEGENERATORDEVICE_CUH
