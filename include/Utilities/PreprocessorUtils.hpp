#ifndef COMPILERUTILS_HPP
#define COMPILERUTILS_HPP

#if __GNUC__ >= 13 && __cplusplus >= 201103L // C++11 or later
    #define STARTCONSTEXPR constexpr
#else
    #define STARTCONSTEXPR
#endif

#if __cplusplus >= 202002L
    #include <numbers>

    #define STARTCONSTINIT constinit
    #define START_PI_NUMBER std::numbers::pi
    #define STARTCONSTEXPRFUNC constexpr
#else
    #define STARTCONSTINIT
    #define STARTCONSTEXPRFUNC

    constexpr double START_PI_NUMBER = 3.14159265358979323846;
#endif
#ifdef __linux__
    #define COMMON_PRETTY_FUNC __PRETTY_FUNCTION__
#elif defined(_WIN32)
    #define COMMON_PRETTY_FUNC __FUNCSIG__
#endif

#endif // !COMPILERUTILS_HPP
