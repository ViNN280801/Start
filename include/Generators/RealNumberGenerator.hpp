#ifndef REALNUMBERGENERATOR_HPP
#define REALNUMBERGENERATOR_HPP

/**
 * @file RealNumberGenerator.hpp
 * @brief Header file for the RealNumberGenerator class, selecting the appropriate implementation for host or device.
 *
 * This file abstracts the implementation of the `RealNumberGenerator` class, automatically
 * selecting the appropriate version based on the target platform. The generator is used
 * for generating random real numbers in a specified range.
 *
 * ### Implementation Details:
 * - **Host Implementation**: Uses `RealNumberGeneratorHost` for CPU computations.
 * - **Device Implementation**: Uses `RealNumberGeneratorDevice` for GPU computations with CUDA.
 *
 * ### Usage:
 * Include this file to use the `RealNumberGenerator` class in a platform-independent way.
 * The correct implementation will be chosen based on the `USE_CUDA` macro.
 */

#ifdef USE_CUDA
    #include "CUDA/RealNumberGeneratorDevice.cuh"         ///< GPU-specific implementation.
    #define RealNumberGenerator RealNumberGeneratorDevice ///< Alias for the device implementation.
#else
    #include "Host/RealNumberGeneratorHost.hpp"         ///< CPU-specific implementation.
    #define RealNumberGenerator RealNumberGeneratorHost ///< Alias for the host implementation.
#endif // !USE_CUDA

#endif // !REALNUMBERGENERATOR_HPP
