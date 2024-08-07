#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <concepts>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <source_location>
#include <sstream>
#include <string_view>

#include "Constants.hpp"
using namespace constants;
using namespace particle_types;

#define STATUS_TO_STR(status) util::getStatusName(status)

#ifdef __linux__
#define COMMON_PRETTY_FUNC __PRETTY_FUNCTION__
#endif
#ifdef _WIN32
#define COMMON_PRETTY_FUNC __FUNCSIG__
#endif

#define ERRMSG_ABS_PATH(desc) std::cerr << std::format("\033[1;31mError:\033[0m\033[1m {}: {}({} line): {}: \033[1;31m{}\033[0m\033[1m\n", \
                                                       util::getCurTime(),                                                                 \
                                                       std::source_location::current().file_name(),                                        \
                                                       std::source_location::current().line(),                                             \
                                                       COMMON_PRETTY_FUNC, desc);
#define LOGMSG_ABS_PATH(desc) std::clog << std::format("Log: {}: {}({} line): {}: {}\n",            \
                                                       util::getCurTime(),                          \
                                                       std::source_location::current().file_name(), \
                                                       std::source_location::current().line(),      \
                                                       COMMON_PRETTY_FUNC, desc);
#define EXTRACT_FILE_NAME(filepath) std::filesystem::path(std::string(filepath).c_str()).filename().string()
#define ERRMSG(desc) std::cerr << std::format("\033[1;31mError:\033[0m\033[1m {}: {}({} line): {}: \033[1;31m{}\033[0m\033[1m\n", \
                                              util::getCurTime(),                                                                 \
                                              EXTRACT_FILE_NAME(std::source_location::current().file_name()),                     \
                                              std::source_location::current().line(),                                             \
                                              COMMON_PRETTY_FUNC, desc);
#define LOGMSG(desc) std::clog << std::format("Log: {}: {}({} line): {}: {}\n",                               \
                                              util::getCurTime(),                                             \
                                              EXTRACT_FILE_NAME(std::source_location::current().file_name()), \
                                              std::source_location::current().line(),                         \
                                              COMMON_PRETTY_FUNC, desc);
#define WARNINGMSG(desc) std::cerr << std::format("\033[1;33mWarning:\033[0m\033[1m {}: {}({} line): {}: {}\n",   \
                                                  util::getCurTime(),                                             \
                                                  EXTRACT_FILE_NAME(std::source_location::current().file_name()), \
                                                  std::source_location::current().line(),                         \
                                                  COMMON_PRETTY_FUNC, desc);

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

    /**
     * @brief Calculates sign.
     * @details Takes a double (`val`) and returns:
     *          - -1 if `val` is less than 0,
     *          -  1 if `val` is greater than 0,
     *          -  0 if `val` is equal to 0.
     */
    constexpr double signFunc(double val)
    {
        if (val < 0)
            return -1;
        if (0 < val)
            return 1;
        return 0;
    }

    /// @brief Helper function to get status name from its value.
    std::string getStatusName(int status);

    /// @brief Helper function to parse and define particle type by string.
    ParticleType getParticleTypeFromStrRepresentation(std::string_view particle);

    /// @brief Helper function to recieve string representation of the particle type.
    std::string getParticleType(ParticleType ptype);

    /**
     * @brief Calculating concentration from the configuration file.
     * @param config Name of the configuration file.
     * @return Concentration. [N] (count).
     * `-1` if smth went wrong.
     */
    double calculateConcentration(std::string_view config);

    /**
     * @brief Checker for file on existence.
     * @param filaname Name of the file (or path) to check it.
     * @return `true` if file exists, otherwise `false`.
     */
    bool exists(std::string_view filename);

    /**
     * @brief Removes file from the PC.
     * @param filename Name of the file (or path) to remove.
     */
    void removeFile(std::string_view filename);

    /// @brief Cheks restrictions for the certain simulation parameters, if something wrong - exits from the program.
    void checkRestrictions(double time_step, size_t particles_count, std::string_view mshfilename);

    /**
     * @brief Checks the validity of a Gmsh mesh file.
     *
     * This function performs several checks on the provided file path to ensure it is a valid Gmsh mesh file.
     * It checks whether the file exists, is not a directory, has the correct ".msh" extension, and is not empty.
     *
     * @param mesh_filename The file path of the Gmsh mesh file to check.
     *
     * @throws std::runtime_error If the file does not exist.
     * @throws std::runtime_error If the provided path is a directory.
     * @throws std::runtime_error If the file extension is not ".msh".
     * @throws std::runtime_error If the file is empty.
     */
    void check_gmsh_mesh_file(std::string_view mesh_filename);
}

#endif // !UTILITIES_HPP
