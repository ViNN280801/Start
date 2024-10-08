#ifndef COMPILERUTILS_HPP
#define COMPILERUTILS_HPP

#if __GNUC__ >= 13 && __cplusplus >= 201103L // C++11 or later
    #define STARTCONSTEXPR constexpr
#else
    #define STARTCONSTEXPR
#endif

#ifdef __linux__
    #define COMMON_PRETTY_FUNC __PRETTY_FUNCTION__
#elif defined(_WIN32)
    #define COMMON_PRETTY_FUNC __FUNCSIG__
#endif

#endif // !COMPILERUTILS_HPP
