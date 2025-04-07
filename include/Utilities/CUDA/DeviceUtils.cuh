#ifndef DEVICE_UTILS_HPP
#define DEVICE_UTILS_HPP

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#include "Utilities/ExceptionMacros.hpp"

START_DEFINE_EXCEPTION(CudaErrorException, std::runtime_error)

#ifdef START_DEBUG
#define START_CHECK_CUDA_ERROR(err, context)                                                              \
    if ((err) != cudaSuccess)                                                                             \
    {                                                                                                     \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << "\n"                                    \
                  << "Context: " << (context) << "\n"                                                     \
                  << "Function: " << __FUNCTION__ << "\n"                                                 \
                  << "File: " << __FILE__ << "\n"                                                         \
                  << "Line: " << __LINE__ << std::endl;                                                   \
        START_THROW_EXCEPTION(CudaErrorException, std::string(context) + ": " + cudaGetErrorString(err)); \
    }
#elif defined(START_RELEASE)
#define START_CHECK_CUDA_ERROR(err, context)                                                              \
    if ((err) != cudaSuccess)                                                                             \
    {                                                                                                     \
        START_THROW_EXCEPTION(CudaErrorException, std::string(context) + ": " + cudaGetErrorString(err)); \
    }
#endif

/**
 * @namespace cuda_utils
 * @brief Provides utility functions for error handling and other CUDA-related operations.
 *
 * The `cuda_utils` namespace contains utility functions to simplify common tasks
 * when working with CUDA. This includes error checking, memory management, and
 * debugging support. These utilities aim to improve code readability and maintainability.
 */
namespace cuda_utils
{
    /**
     * @brief Checks the result of a CUDA operation and throws an exception if an error occurred.
     *
     * This function simplifies error handling for CUDA API calls and kernel execution.
     * If the passed error code indicates a failure, the function throws a `std::runtime_error`
     * with a descriptive message that includes the provided context and the CUDA error string.
     *
     * Example usage:
     * @code
     * cudaError_t err = cudaMalloc(&d_data, size);
     * cuda_utils::checkCudaError(err, "Failed to allocate device memory");
     * @endcode
     *
     * @param err The CUDA error code to check. It is expected to be the result of a CUDA API call
     *            or `cudaGetLastError` after a kernel execution.
     * @param message A context-specific message describing the operation where the error occurred.
     *                This message is included in the exception if an error is detected.
     *
     * @throws std::runtime_error if the error code is not `cudaSuccess`.
     */
    inline void check_cuda_err(cudaError_t err, char const *message)
    {
        if (err != cudaSuccess)
            START_THROW_EXCEPTION(CudaErrorException, std::string(message) + ": " + cudaGetErrorString(err));
    }
}

#endif // !USE_CUDA
#endif // !DEVICE_UTILS_HPP
