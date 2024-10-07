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
    #define ERRMSG_ABS_PATH(desc) std::cerr << util::stringify("\033[1;31mError:\033[0m\033[1m ",                       \
                                                            getCurTime_Impl(),                                       \
                                                            ": ",                                                    \
                                                            std::source_location::current().file_name(),             \
                                                            "(", std::source_location::current().line(), " line): ", \
                                                            COMMON_PRETTY_FUNC, ": \033[1;31m", desc, "\033[0m\033[1m\n");
    #define LOGMSG_ABS_PATH(desc) std::clog << util::stringify("Log: ", getCurTime_Impl(), ": ",                        \
                                                            std::source_location::current().file_name(),             \
                                                            "(", std::source_location::current().line(), " line): ", \
                                                            COMMON_PRETTY_FUNC, ": ", desc, "\n");
    #define EXTRACT_FILE_NAME(filepath) std::filesystem::path(std::string(filepath).c_str()).filename().string()
#endif

#ifdef SHOW_LOGS
    #define ERRMSGSTR(desc) util::stringify("\033[1;31mError:\033[0m\033[1m ", getCurTime_Impl(),                 \
                                            ": ", EXTRACT_FILE_NAME(std::source_location::current().file_name()), \
                                            "(", std::source_location::current().line(), " line): ",              \
                                            COMMON_PRETTY_FUNC, ": \033[1;31m", desc, "\033[0m\033[1m\n");
    #define ERRMSG(desc) std::cerr << ERRMSGSTR(desc);
#else
    #define ERRMSG(desc)
#endif

#ifdef SHOW_LOGS
    #define LOGMSGSTR(desc) util::stringify("Log: ", getCurTime_Impl(), ": ",                               \
                                            EXTRACT_FILE_NAME(std::source_location::current().file_name()), \
                                            "(", std::source_location::current().line(), " line): ",        \
                                            COMMON_PRETTY_FUNC, ": ", desc, "\n");
    #define LOGMSG(desc) std::clog << LOGMSGSTR(desc);
#else
    #define LOGMSG(desc)
#endif

#ifdef SHOW_LOGS
    #define WARNINGMSGSTR(desc) util::stringify("\033[1;33mWarning:\033[0m\033[1m ", getCurTime_Impl(),               \
                                                ": ", EXTRACT_FILE_NAME(std::source_location::current().file_name()), \
                                                "(", std::source_location::current().line(), " line): ",              \
                                                COMMON_PRETTY_FUNC, ": ", desc, "\n");
    #define WARNINGMSG(desc) std::cerr << WARNINGMSGSTR(desc);
#else
    #define WARNINGMSG(desc)
#endif

inline std::string getCurTime_Impl(std::string_view format = "%H:%M:%S")
{
    std::chrono::system_clock::time_point tp{std::chrono::system_clock::now()};
    time_t tt{std::chrono::system_clock::to_time_t(tp)};
    tm *t{localtime(&tt)};
    std::stringstream ss;
    ss << std::put_time(t, std::string(format).c_str());
    return ss.str();
}

#endif // !LOGMACROS_HPP
