#ifndef PREPROCESSOR_UTILS_HPP
#define PREPROCESSOR_UTILS_HPP

#include "Utilities/VersionCheck.hpp"

/**
 * @file PreprocessorUtils.hpp
 * @brief Utility macros for preprocessor directives
 * 
 * This file contains utility macros for preprocessor directives,
 * including platform-specific and compiler-specific macros.
 */

// Platform detection
#if defined(_WIN32) || defined(_WIN64)
    #define START_PLATFORM_WINDOWS 1
#elif defined(__linux__)
    #define START_PLATFORM_LINUX 1
#elif defined(__APPLE__)
    #define START_PLATFORM_MACOS 1
#else
    #define START_PLATFORM_UNKNOWN 1
#endif

// Compiler detection
#if defined(_MSC_VER)
    #define START_COMPILER_MSVC 1
#elif defined(__GNUC__)
    #define START_COMPILER_GCC 1
#elif defined(__clang__)
    #define START_COMPILER_CLANG 1
#else
    #define START_COMPILER_UNKNOWN 1
#endif

// Architecture detection
#if defined(__x86_64__) || defined(_M_X64)
    #define START_ARCH_X64 1
#elif defined(__i386) || defined(_M_IX86)
    #define START_ARCH_X86 1
#elif defined(__arm__) || defined(_M_ARM)
    #define START_ARCH_ARM 1
#elif defined(__aarch64__)
    #define START_ARCH_ARM64 1
#else
    #define START_ARCH_UNKNOWN 1
#endif

// Debug/Release detection
#if defined(DEBUG) || defined(_DEBUG)
    #define START_DEBUG 1
#else
    #define START_RELEASE 1
#endif

// C++ version specific macros
#if __GNUC__ >= 12 && __cplusplus >= 201103L // C++11 or later
    #define STARTCONSTEXPR constexpr
#else
    #define STARTCONSTEXPR
#endif

#if __GNUC__ >= 12 && __cplusplus >= 202002L
    #include <numbers>

    #define STARTCONSTINIT constinit
    #define START_PI_NUMBER std::numbers::pi
    #define STARTCONSTEXPRFUNC constexpr
#elif __cplusplus >= 201103L
    #define STARTCONSTINIT
    #define STARTCONSTEXPRFUNC

    constexpr double START_PI_NUMBER = 3.14159265358979323846;
#else
    #define STARTCONSTINIT
    #define STARTCONSTEXPRFUNC
    #define START_PI_NUMBER 3.14159265358979323846
#endif

// Platform-specific function name macros
#ifdef __linux__
    #define COMMON_PRETTY_FUNC __PRETTY_FUNCTION__
#elif defined(_WIN32)
    #define COMMON_PRETTY_FUNC __FUNCSIG__
#endif

// CUDA-specific macros
#ifdef USE_CUDA
    #define START_CUDA_HOST_DEVICE __host__ __device__
    #define START_CUDA_HOST __host__
    #define START_CUDA_GLOBAL extern "C" __global__
    #define START_CUDA_GLOBAL_EXTERN_C extern "C" __global__
    #define START_CUDA_DEVICE_EXTERN_C extern "C" __device__
#else
    #define START_CUDA_HOST_DEVICE
    #define START_CUDA_HOST
    #define START_CUDA_GLOBAL
    #define START_CUDA_DEVICE_EXTERN_C
#endif

#if defined(USE_CUDA) && defined(__CUDA_ARCH__)
    #define START_CUDA_DEVICE __device__
#else
    #define START_CUDA_DEVICE
#endif

// Utility macros
#define START_STRINGIFY(x) #x
#define START_TOSTRING(x) START_STRINGIFY(x)
#define START_CONCAT(a, b) a##b

// Function attributes
#if defined(START_COMPILER_GCC) || defined(START_COMPILER_CLANG)
    #define START_FORCE_INLINE __attribute__((always_inline)) inline
    #define START_NO_INLINE __attribute__((noinline))
    #define START_DEPRECATED(msg) __attribute__((deprecated(msg)))
#elif defined(START_COMPILER_MSVC)
    #define START_FORCE_INLINE __forceinline
    #define START_NO_INLINE __declspec(noinline)
    #define START_DEPRECATED(msg) __declspec(deprecated(msg))
#else
    #define START_FORCE_INLINE inline
    #define START_NO_INLINE
    #define START_DEPRECATED(msg)
#endif

// Branch prediction hints (GCC/Clang only)
#if defined(START_COMPILER_GCC) || defined(START_COMPILER_CLANG)
    #define START_LIKELY(x) __builtin_expect(!!(x), 1)
    #define START_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
    #define START_LIKELY(x) (x)
    #define START_UNLIKELY(x) (x)
#endif

#endif // !PREPROCESSOR_UTILS_HPP
