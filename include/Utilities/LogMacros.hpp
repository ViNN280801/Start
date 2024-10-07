#ifndef LOGMACROS_HPP
#define LOGMACROS_HPP

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <source_location>
#include <sstream>
#include <string_view>

#include "Utilities/PreprocessorUtils.hpp"

#ifdef SHOW_LOGS
    #define ERRMSG_ABS_PATH(desc) std::cerr << util::stringify("\033[1;31mError:\033[0m\033[1m ",                    \
                                                            util::getCurTime(),                                      \
                                                            ": ",                                                    \
                                                            std::source_location::current().file_name(),             \
                                                            "(", std::source_location::current().line(), " line): ", \
                                                            COMMON_PRETTY_FUNC, ": \033[1;31m", desc, "\033[0m\033[1m\n");
    #define LOGMSG_ABS_PATH(desc) std::clog << util::stringify("Log: ", util::getCurTime(), ": ",                    \
                                                            std::source_location::current().file_name(),             \
                                                            "(", std::source_location::current().line(), " line): ", \
                                                            COMMON_PRETTY_FUNC, ": ", desc, "\n");
    #define EXTRACT_FILE_NAME(filepath) std::filesystem::path(std::string(filepath).c_str()).filename().string()
#endif

#ifdef SHOW_LOGS
    #define ERRMSGSTR(desc) util::stringify("\033[1;31mError:\033[0m\033[1m ", util::getCurTime(),                \
                                            ": ", EXTRACT_FILE_NAME(std::source_location::current().file_name()), \
                                            "(", std::source_location::current().line(), " line): ",              \
                                            COMMON_PRETTY_FUNC, ": \033[1;31m", desc, "\033[0m\033[1m\n");
    #define ERRMSG(desc) std::cerr << ERRMSGSTR(desc);
#else
    #define ERRMSG(desc)
#endif

#ifdef SHOW_LOGS
    #define LOGMSGSTR(desc) util::stringify("Log: ", util::getCurTime(), ": ",                              \
                                            EXTRACT_FILE_NAME(std::source_location::current().file_name()), \
                                            "(", std::source_location::current().line(), " line): ",        \
                                            COMMON_PRETTY_FUNC, ": ", desc, "\n");
    #define LOGMSG(desc) std::clog << LOGMSGSTR(desc);
#else
    #define LOGMSG(desc)
#endif

#ifdef SHOW_LOGS
    #define WARNINGMSGSTR(desc) util::stringify("\033[1;33mWarning:\033[0m\033[1m ", util::getCurTime(),              \
                                                ": ", EXTRACT_FILE_NAME(std::source_location::current().file_name()), \
                                                "(", std::source_location::current().line(), " line): ",              \
                                                COMMON_PRETTY_FUNC, ": ", desc, "\n");
    #define WARNINGMSG(desc) std::cerr << WARNINGMSGSTR(desc);
#else
    #define WARNINGMSG(desc)
#endif

#define DEVELOPER_MAIL "vladislav_semykin01@mail.ru"
#define CONTACT_SUPPORT_MSG(desc) util::stringify("Internal Error: ", desc, ". Contact support: ", DEVELOPER_MAIL, "\n");
#define CONTACT_SUPPORT_MSG_TO_CERR(desc) std::cerr << CONTACT_SUPPORT_MSG(desc);

namespace util
{
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

    /**
     * @brief Gets the current system time in the specified format.
     * @tparam Format A format string compatible with std::put_time.
     * Defaults to "%H:%M:%S" if not specified.
     * For example, "%Y-%m-%d %H:%M:%S" for date and time in YYYY-MM-DD HH:MM:SS format.
     * @param format The format string compatible with std::put_time. Defaults to "%H:%M:%S".
     */
    std::string getCurTime(std::string_view format = "%H:%M:%S");

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
}

#endif // !LOGMACROS_HPP
