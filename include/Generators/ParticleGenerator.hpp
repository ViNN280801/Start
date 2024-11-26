#ifndef PARTICLEGENERATOR_HPP
#define PARTICLEGENERATOR_HPP

/**
 * @file ParticleGenerator.hpp
 * @brief Header file for the ParticleGenerator class, selecting the appropriate implementation for host or device.
 *
 * This file provides an abstraction for the `ParticleGenerator` class, ensuring the correct
 * implementation is used based on the target platform. The generator creates particles
 * based on source specifications (e.g., point or surface sources).
 *
 * ### Implementation Details:
 * - **Host Implementation**: Uses `ParticleGeneratorHost` for CPU-based particle generation.
 * - **Device Implementation**: Uses `ParticleGeneratorDevice` for GPU-based particle generation with CUDA.
 *
 * ### Usage:
 * Include this file to use the `ParticleGenerator` class without worrying about platform specifics.
 * The correct implementation will be chosen automatically based on the `USE_CUDA` macro.
 */

#ifdef USE_CUDA
    #include "CUDA/ParticleGeneratorDevice.cuh"       ///< GPU-specific implementation.
    #define ParticleGenerator ParticleGeneratorDevice ///< Alias for the device implementation.
#else
    #include "Host/ParticleGeneratorHost.hpp"       ///< CPU-specific implementation.
    #define ParticleGenerator ParticleGeneratorHost ///< Alias for the host implementation.
#endif // !USE_CUDA

#endif // !PARTICLEGENERATOR_HPP
