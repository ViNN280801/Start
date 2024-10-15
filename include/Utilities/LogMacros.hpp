#ifndef LOGMACROS_HPP
#define LOGMACROS_HPP

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string_view>

#if __cplusplus >= 202002L
    #include <source_location>
    using SourceLocation = std::source_location;
#else
    struct SourceLocation {
        static constexpr SourceLocation current() noexcept { return {}; }
        constexpr const char* file_name() const noexcept { return "NOT FOR THE RELEASE!!! unknown file"; }
        constexpr int line() const noexcept { return -1; }
    };
#endif

#include "Utilities/PreprocessorUtils.hpp"

#ifdef SHOW_LOGS
    #define ERRMSG_ABS_PATH(desc) std::cerr << util::stringify("\033[1;31mError:\033[0m\033[1m ",                    \
                                                            util::getCurTime(),                                      \
                                                            ": ",                                                    \
                                                            SourceLocation::current().file_name(),             \
                                                            "(", SourceLocation::current().line(), " line): ", \
                                                            COMMON_PRETTY_FUNC, ": \033[1;31m", desc, "\033[0m\033[1m\n");
    #define LOGMSG_ABS_PATH(desc) std::clog << util::stringify("Log: ", util::getCurTime(), ": ",                    \
                                                            SourceLocation::current().file_name(),             \
                                                            "(", SourceLocation::current().line(), " line): ", \
                                                            COMMON_PRETTY_FUNC, ": ", desc, "\n");
    #define EXTRACT_FILE_NAME(filepath) std::filesystem::path(std::string(filepath).c_str()).filename().string()
#endif

#ifdef SHOW_LOGS
    #ifdef START_RELEASE
        #define ERRMSGSTR(desc) util::stringify("\033[1;31mError:\033[0m\033[1m ", util::getCurTime(), \
                                                        ": ", desc, "\n");
        #define ERRMSG(desc) std::cerr << ERRMSGSTR(desc);
    #else
        #define ERRMSGSTR(desc) util::stringify("\033[1;31mError:\033[0m\033[1m ", util::getCurTime(),            \
                                            ": ", EXTRACT_FILE_NAME(SourceLocation::current().file_name()), \
                                            "(", SourceLocation::current().line(), " line): ",              \
                                            COMMON_PRETTY_FUNC, ": \033[1;31m", desc, "\033[0m\033[1m\n");
        #define ERRMSG(desc) std::cerr << ERRMSGSTR(desc);
    #endif
#else
    #define ERRMSG(desc)
#endif

#ifdef SHOW_LOGS    
    #ifdef START_RELEASE
        #define LOGMSGSTR(desc) util::stringify("Log: ", util::getCurTime(), \
                                                        ": ", desc, "\n");
        #define LOGMSG(desc) std::clog << LOGMSGSTR(desc);
    #else
        #define LOGMSGSTR(desc) util::stringify("Log: ", util::getCurTime(), ": ",                          \
                                            EXTRACT_FILE_NAME(SourceLocation::current().file_name()), \
                                            "(", SourceLocation::current().line(), " line): ",        \
                                            COMMON_PRETTY_FUNC, ": ", desc, "\n");
        #define LOGMSG(desc) std::clog << LOGMSGSTR(desc);
    #endif
#else
    #define LOGMSG(desc)
#endif

#ifdef SHOW_LOGS
    #ifdef START_RELEASE
        #define WARNINGMSGSTR(desc) util::stringify("\033[1;33mWarning:\033[0m\033[1m ", util::getCurTime(), \
                                                        ": ", desc, "\n");
        #define WARNINGMSG(desc) std::cerr << WARNINGMSGSTR(desc);
    #else
        #define WARNINGMSGSTR(desc) util::stringify("\033[1;33mWarning:\033[0m\033[1m ", util::getCurTime(),          \
                                                ": ", EXTRACT_FILE_NAME(SourceLocation::current().file_name()), \
                                                "(", SourceLocation::current().line(), " line): ",              \
                                                COMMON_PRETTY_FUNC, ": ", desc, "\n");
        #define WARNINGMSG(desc) std::cerr << WARNINGMSGSTR(desc);
    #endif
#else
    #define WARNINGMSG(desc)
#endif

#ifdef SHOW_LOGS
    #ifdef START_RELEASE
        #define SUCCESSMSGSTR(desc) util::stringify("\033[1;32mSuccess:\033[0m\033[1m ", util::getCurTime(), \
                                                        ": ", desc, "\n");
        #define SUCCESSMSG(desc) std::cerr << SUCCESSMSGSTR(desc);
    #else
        #define SUCCESSMSGSTR(desc) util::stringify("\033[1;32mSuccess:\033[0m\033[1m ", util::getCurTime(),          \
                                                ": ", EXTRACT_FILE_NAME(SourceLocation::current().file_name()), \
                                                "(", SourceLocation::current().line(), " line): ",              \
                                                COMMON_PRETTY_FUNC, ": ", desc, "\n");
        #define SUCCESSMSG(desc) std::cerr << SUCCESSMSGSTR(desc);
    #endif
#else
    #define SUCCESSMSG(desc)
#endif

namespace util
{
#if __cplusplus >= 202002L
    /**
     * @brief Concept that specifies all types that can be convert to "std::string_view" type
     * For example, "char", "const char *", "std::string", etc.
     * @tparam T The type to check for convertibility to std::string_view.
     */
    template <typename T>
    concept StringConvertible = std::is_convertible_v<T, std::string_view>;

    /**
     * @brief Concept that checks if variable has output operator
     * @tparam a variable to check
     * @param os output stream
     */
    template <typename T>
    concept Printable = requires(T a, std::ostream &os) {
        {
            os << a
        } -> std::same_as<std::ostream &>;
    };
#else
    // Fallback implementation for C++17 using SFINAE
    template <typename T>
    using StringConvertible = std::enable_if_t<std::is_convertible_v<T, std::string_view>>;

    template <typename T>
    using Printable = std::enable_if_t<std::is_same_v<decltype(std::declval<std::ostream&>() << std::declval<T>()), std::ostream&>>;
#endif

    /**
     * @brief Gets the current system time in the specified format.
     * @tparam Format A format string compatible with std::put_time.
     * Defaults to "%H:%M:%S" if not specified.
     * For example, "%Y-%m-%d %H:%M:%S" for date and time in YYYY-MM-DD HH:MM:SS format.
     * @param format The format string compatible with std::put_time. Defaults to "%H:%M:%S".
     */
    std::string getCurTime(std::string_view format = "%H:%M:%S");

#if __cplusplus >= 202002L
    /**
     * @brief Generates string with specified multiple args
     * @tparam args arguments of type that can be convert to string
     * @return String composed from all arguments
     */
    template <Printable... Args>
    std::string stringify(Args &&...args)
    {
        std::ostringstream oss;
        (oss << ... << std::forward<Args>(args));
        return oss.str();
    }
#else
    template <typename... Args>
    std::string stringify(Args &&...args)
    {
        std::ostringstream oss;
        (oss << ... << args);
        return oss.str();
    }
#endif
}

#endif // !LOGMACROS_HPP
