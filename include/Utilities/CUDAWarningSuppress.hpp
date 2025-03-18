#ifndef CUDA_WARNING_SUPPRESS_HPP
#define CUDA_WARNING_SUPPRESS_HPP

/**
 * @file CUDAWarningSuppress.hpp
 * @brief Macros for suppressing CUDA warnings
 * 
 * This file contains macros that help suppress warnings that occur when using CUDA
 * with libraries like Boost and CGAL that are not fully CUDA-compatible.
 */

#ifdef USE_CUDA
// Macro to suppress warnings about __host__ and __device__ annotations on defaulted functions
#define START_CUDA_SUPPRESS_HOST_DEVICE_WARNING _Pragma("nv_diag_suppress 20012")
#define START_CUDA_RESTORE_HOST_DEVICE_WARNING _Pragma("nv_diag_default 20012")

// Macro to suppress warnings about calling __host__ functions from __host__ __device__ functions
#define START_CUDA_SUPPRESS_HOST_CALL_WARNING _Pragma("nv_diag_suppress 20014")
#define START_CUDA_RESTORE_HOST_CALL_WARNING _Pragma("nv_diag_default 20014")

// Macro to suppress all CUDA warnings in a section of code
#define START_CUDA_SUPPRESS_ALL_WARNINGS \
    START_CUDA_SUPPRESS_HOST_DEVICE_WARNING \
    START_CUDA_SUPPRESS_HOST_CALL_WARNING

// Macro to restore all CUDA warnings
#define START_CUDA_RESTORE_ALL_WARNINGS \
    START_CUDA_RESTORE_HOST_DEVICE_WARNING \
    START_CUDA_RESTORE_HOST_CALL_WARNING

// Macros used in the codebase for warning suppression
#define START_CUDA_WARNING_SUPPRESS START_CUDA_SUPPRESS_ALL_WARNINGS
#define END_CUDA_WARNING_SUPPRESS START_CUDA_RESTORE_ALL_WARNINGS


#else
// Define empty macros when CUDA is not used
#define START_CUDA_SUPPRESS_HOST_DEVICE_WARNING
#define START_CUDA_RESTORE_HOST_DEVICE_WARNING
#define START_CUDA_SUPPRESS_HOST_CALL_WARNING
#define START_CUDA_RESTORE_HOST_CALL_WARNING
#define START_CUDA_SUPPRESS_ALL_WARNINGS
#define START_CUDA_RESTORE_ALL_WARNINGS
#define START_CUDA_WARNING_SUPPRESS
#define END_CUDA_WARNING_SUPPRESS
#define START_CUDA_HOST_DEVICE
#endif

#endif // !CUDA_WARNING_SUPPRESS_HPP
