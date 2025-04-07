#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include "Utilities/Constants.hpp"
#include "Utilities/LogMacros.hpp"
#include "Utilities/Types.hpp"
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
     * @brief Calculating concentration from the configuration file with printing error msg,
     *        when calculated gas concentration is too small according to the constants.
     * @param config Name of the configuration file.
     * @return Concentration. [N] (count).
     * `-1` if smth went wrong.
     */
    double calculateConcentration(std::string_view config);

    /**
     * @brief Calculating concentration from the configuration file with printing error msg,
     *        when calculated gas concentration is too small according to the constants.
     * @param pressure Pressure. [Pa].
     * @param temperature Temperature. [K].
     * @return Concentration. [N] (count).
     * `-1` if smth went wrong.
     */
    double calculateConcentration(double pressure, double temperature);

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

    /**
     * @brief Converts energy from electron volts (eV) to joules (J) in place.
     *
     * This function modifies the input energy value, converting it from eV to J,
     * using the predefined physical constant `constants::physical_constants::eV_J`.
     *
     * The conversion formula is:
     * \f$ E_{J} = E_{eV} \cdot eV\_J \f$
     *
     * @param[in,out] energy_eV Reference to the energy value in electron volts (eV).
     *                          The value is updated to its equivalent in joules (J).
     */
    START_CUDA_HOST_DEVICE inline void convert_energy_eV_to_energy_J_inplace(double &energy_eV) { energy_eV *= constants::physical_constants::eV_J; }

    /**
     * @brief Converts energy from joules (J) to electron volts (eV) in place.
     *
     * This function modifies the input energy value, converting it from J to eV,
     * using the predefined physical constant `constants::physical_constants::J_eV`.
     *
     * The conversion formula is:
     * \f$ E_{eV} = E_{J} \cdot J\_eV \f$
     *
     * @param[in,out] energy_J Reference to the energy value in joules (J).
     *                          The value is updated to its equivalent in electron volts (eV).
     */
    START_CUDA_HOST_DEVICE inline void convert_energy_J_to_energy_eV_inplace(double &energy_J) { energy_J *= constants::physical_constants::J_eV; }

    /**
     * @brief Converts energy from electron volts (eV) to joules (J).
     *
     * This function performs the conversion of energy values from eV to J using the
     * predefined physical constant `constants::physical_constants::eV_J`.
     *
     * The conversion formula is:
     * \f$ E_{J} = E_{eV} \cdot eV\_J \f$
     *
     * @param energy_eV Energy value in electron volts (eV).
     * @return Energy value in joules (J).
     */
    START_CUDA_HOST_DEVICE inline double convert_energy_eV_to_energy_J(double energy_eV) { return energy_eV * constants::physical_constants::eV_J; }

    /**
     * @brief Converts energy from joules (J) to electron volts (eV).
     *
     * This function performs the conversion of energy values from J to eV using the
     * predefined physical constant `constants::physical_constants::J_eV`.
     *
     * The conversion formula is:
     * \f$ E_{eV} = E_{J} \cdot J\_eV \f$
     *
     * @param energy_J Energy value in joules (J).
     * @return Energy value in electron volts (eV).
     */
    START_CUDA_HOST_DEVICE inline double convert_energy_J_to_energy_eV(double energy_J) { return energy_J * constants::physical_constants::J_eV; }
}

#endif // !UTILITIES_HPP
