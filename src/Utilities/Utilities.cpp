#include <chrono>
#include <nlohmann/json.hpp>
#include <sstream>
#include <sys/stat.h>

#include "Utilities/ConfigParser.hpp"
#include "Utilities/Utilities.hpp"
#include "Utilities/UtilitiesExceptions.hpp"

using json = nlohmann::json;

std::string util::getCurTime(std::string_view format)
{
    std::chrono::system_clock::time_point tp{std::chrono::system_clock::now()};
    time_t tt{std::chrono::system_clock::to_time_t(tp)};
    tm *t{localtime(&tt)};
    std::stringstream ss;
    ss << std::put_time(t, std::string(format).c_str());
    return ss.str();
}

std::string util::getStatusName(int status)
{
    switch (status)
    {
    case -15:
        return "BAD_MSHFILE";
    case -14:
        return "JSON_BAD_PARSE";
    case -13:
        return "JSON_BAD_PARAM";
    case -12:
        return "BAD_PARTICLE_COUNT";
    case -11:
        return "BAD_THREAD_COUNT";
    case -10:
        return "BAD_TIME_STEP";
    case -9:
        return "BAD_SIMTIME";
    case -8:
        return "BAD_VOLUME";
    case -7:
        return "BAD_PRESSURE";
    case -6:
        return "BAD_TEMPERATURE";
    case -5:
        return "BAD_ENERGY";
    case -4:
        return "BAD_MODEL";
    case -3:
        return "UNKNOWN_PARTICLES";
    case -2:
        return "BAD_PARTICLES_FORMAT";
    case -1:
        return "BAD_FILE";
    case 0:
        return "EMPTY_STR";
    case 1:
        return "STATUS_OK";
    default:
        return "UNKNOWN_ERROR";
    }
}

ParticleType util::getParticleTypeFromStrRepresentation(std::string_view particle)
{
    if (particle == "O2")
        return O2;
    else if (particle == "Ar")
        return Ar;
    else if (particle == "Ne")
        return Ne;
    else if (particle == "He")
        return He;
    else if (particle == "Ti")
        return Ti;
    else if (particle == "Al")
        return Al;
    else if (particle == "Sn")
        return Sn;
    else if (particle == "W")
        return W;
    else if (particle == "Au")
        return Au;
    else if (particle == "Cu")
        return Cu;
    else if (particle == "Ni")
        return Ni;
    else if (particle == "Ag")
        return Ag;
    else
        return Unknown;
}

std::string util::getParticleType(ParticleType ptype)
{
    switch (ptype)
    {
    case O2:
        return "O2";
    case Ar:
        return "Ar";
    case Ne:
        return "Ne";
    case He:
        return "He";
    case Ti:
        return "Ti";
    case Al:
        return "Al";
    case Sn:
        return "Sn";
    case W:
        return "W";
    case Au:
        return "Au";
    case Ni:
        return "Ni";
    case Ag:
        return "Ag";
    default:
        return "Unknown";
    }
}

double util::calculateConcentration(std::string_view config)
{
    ConfigParser parser(config);
    return calculateConcentration(parser.getPressure(), parser.getTemperature());
}

double util::calculateConcentration(double pressure, double temperature)
{
    // PV = nRT: https://en.wikipedia.org/wiki/Ideal_gas_law.
    double gasConcentration{(pressure / (constants::physical_constants::R * temperature)) *
                            constants::physical_constants::N_av};
    if (gasConcentration < constants::gasConcentrationMinimalValue)
    {
        WARNINGMSG(util::stringify("Something wrong with the concentration of the gas. Its value is ",
                                   gasConcentration, ". Simulation might considerably slows down"));
    }
    if (gasConcentration < 0)
    {
        START_THROW_EXCEPTION(UtilsZeroOrNegativeGasConcentrationException,
                              util::stringify("Failed to calculate concentration of the gas. Its value is less than 0. Pressure: ",
                                              pressure, ". Temperature: ", temperature));
    }
    return gasConcentration;
}

bool util::exists(std::string_view filename)
{
    struct stat buf;
    return (stat(filename.data(), std::addressof(buf)) == 0);
}

void util::removeFile(std::string_view filename) { std::filesystem::remove(filename); }

void util::checkRestrictions(double time_step, size_t particles_count, std::string_view mshfilename)
{
    if (not util::exists(mshfilename))
    {
        ERRMSG(util::stringify("File (", mshfilename, ") doesn't exist"));
        std::exit(EXIT_FAILURE);
    }
    if (time_step <= 0.0)
    {
        ERRMSG(util::stringify("Time step can't be less or equal 0"));
        std::exit(EXIT_FAILURE);
    }
    if (particles_count > 10'000'000)
    {
        ERRMSG(util::stringify("Particles count limited by 10'000'000.\nBut you entered ", particles_count));
        std::exit(EXIT_FAILURE);
    }
}

void util::check_json_validity(std::string_view json_filename)
{
    // Verify JSON file content after saving.
    std::ifstream infile(json_filename.data());
    json loadedJson;
    if (infile.is_open())
    {
        infile >> loadedJson;
        infile.close();

        // Check if the file contains null or is incorrectly formatted.
        if (loadedJson.is_null() || loadedJson.empty() || !loadedJson.is_object())
            START_THROW_EXCEPTION(UtilsInvalidJSONFileException,
                                  util::stringify("Json file '", json_filename, "' is invalid. Check the formatting or content."));
    }
    else
        START_THROW_EXCEPTION(UtilsFailedToOpenFileException,
                              util::stringify("Failed to open file for reading back and check its content."));
}
