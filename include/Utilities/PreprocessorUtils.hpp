#ifndef PREPROCESSORUTILS_HPP
#define PREPROCESSORUTILS_HPP

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

#ifdef __linux__
    #define COMMON_PRETTY_FUNC __PRETTY_FUNCTION__
#elif defined(_WIN32)
    #define COMMON_PRETTY_FUNC __FUNCSIG__
#endif

#ifdef USE_CUDA
    #define START_CUDA_HOST_DEVICE __host__ __device__
    #define START_CUDA_HOST __host__
    #define START_CUDA_GLOBAL __global__
#else
    #define START_CUDA_HOST_DEVICE
    #define START_CUDA_HOST
    #define START_CUDA_GLOBAL
#endif

#if defined(USE_CUDA) && defined(__CUDA_ARCH__)
    #define START_CUDA_DEVICE __device__
#else
    #define START_CUDA_DEVICE
#endif

#endif // !PREPROCESSORUTILS_HPP
