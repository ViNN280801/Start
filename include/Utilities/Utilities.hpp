#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#if __cplusplus >= 202002L
#include <concepts>
#else
#include <type_traits>
#endif

#include "Utilities/Constants.hpp"
#include "Utilities/LogMacros.hpp"
using namespace constants;
using namespace particle_types;

#define STATUS_TO_STR(status) util::getStatusName(status)
#define UNKNOWN_BUILD_CONFIGURATION "Unknown build configuration"

namespace util
{
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

    /**
     * @brief Validates the contents of a JSON file.
     *
     * This function reads the JSON file specified by the given filename and checks its validity.
     * The function ensures that the file can be opened, parsed, and contains valid JSON data.
     * If the file is not found, is not correctly formatted, or contains invalid data, the function
     * throws an exception.
     *
     * @param json_filename The name of the JSON file to validate.
     *
     * @throws std::runtime_error If the JSON file is null, empty, or incorrectly formatted.
     * @throws std::ios_base::failure If the file cannot be opened for reading.
     *
     * Example Usage:
     * @code
     * try
     * {
     *     util::check_json_validity("config.json");
     * }
     * catch (const std::exception &e)
     * {
     *     std::cerr << "Error: " << e.what() << std::endl;
     * }
     * @endcode
     */
    void check_json_validity(std::string_view json_filename);
}

#endif // !UTILITIES_HPP
