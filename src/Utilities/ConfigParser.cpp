#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include "Utilities/ConfigParser.hpp"
#include "Utilities/UtilitiesExceptions.hpp"

void checkParameterExists(json const &j, std::string_view param)
{
    if (!j.contains(param.data()))
        START_THROW_EXCEPTION(UtilsMissingRequiredParameterException,
                              util::stringify("Missing required parameter: ",
                                              param, ". Example: \"", param, "\": <value>"));
}

void ConfigParser::_processRequiredParameters(json const &configJson)
{
    // Check for all required parameters in the JSON file
    checkParameterExists(configJson, "Mesh File");
    checkParameterExists(configJson, "Threads");
    checkParameterExists(configJson, "Time Step");
    checkParameterExists(configJson, "Simulation Time");
    checkParameterExists(configJson, "T");
    checkParameterExists(configJson, "P");
    checkParameterExists(configJson, "Gas");
    checkParameterExists(configJson, "Model");
    checkParameterExists(configJson, "EdgeSize");
    checkParameterExists(configJson, "DesiredAccuracy");
}

void ConfigParser::_processSimulationParameters(json const &configJson)
{
    m_config.mshfilename = configJson.at("Mesh File").get<std::string>();
    m_config.num_threads = configJson.at("Threads").get<unsigned int>();
    m_config.time_step = configJson.at("Time Step").get<double>();
    m_config.simtime = configJson.at("Simulation Time").get<double>();
    m_config.temperature = configJson.at("T").get<double>();
    m_config.pressure = configJson.at("P").get<double>();
    m_config.gas = util::getParticleTypeFromStrRepresentation(configJson.at("Gas").get<std::string>());
    m_config.model = configJson.at("Model").get<std::string>();

    // Process optional sputtering parameter
    if (configJson.contains("Sputtering"))
        m_config.sputtering = configJson.at("Sputtering").get<bool>();

    // Check if mesh file exists
    std::ifstream mesh_file(m_config.mshfilename);
    if (!mesh_file.good())
    {
        std::cerr << "Warning: Mesh file '" << m_config.mshfilename << "' does not exist or cannot be accessed." << std::endl;
        WARNINGMSG(util::stringify("Warning: Mesh file '", m_config.mshfilename, "' does not exist or cannot be accessed."));
    }
    mesh_file.close();
}

void ConfigParser::_processPointParticleSources(json const &configJson)
{
    if (!configJson.contains("ParticleSourcePoint"))
        return;

    for (auto &[key, particleSource] : configJson.at("ParticleSourcePoint").items())
    {
        checkParameterExists(particleSource, "Type");
        checkParameterExists(particleSource, "Count");
        checkParameterExists(particleSource, "Energy");
        checkParameterExists(particleSource, "phi");
        checkParameterExists(particleSource, "theta");
        checkParameterExists(particleSource, "expansionAngle");
        checkParameterExists(particleSource, "BaseCoordinates");

        point_source_t sourcePoint;
        sourcePoint.type = particleSource.at("Type").get<std::string>();
        sourcePoint.count = particleSource.at("Count").get<size_t>();
        sourcePoint.energy = particleSource.at("Energy").get<double>();
        sourcePoint.phi = particleSource.at("phi").get<double>();
        sourcePoint.theta = particleSource.at("theta").get<double>();
        sourcePoint.expansionAngle = particleSource.at("expansionAngle").get<double>();
        sourcePoint.baseCoordinates = particleSource.at("BaseCoordinates").get<std::array<double, 3ul>>();

        m_config.particleSourcePoints.push_back(sourcePoint);
    }
}

void ConfigParser::_processSurfaceParticleSources(json const &configJson)
{
    if (!configJson.contains("ParticleSourceSurface"))
        return;

    for (auto &[key, particleSource] : configJson.at("ParticleSourceSurface").items())
    {
        checkParameterExists(particleSource, "Type");
        checkParameterExists(particleSource, "Count");
        checkParameterExists(particleSource, "Energy");
        checkParameterExists(particleSource, "BaseCoordinates");

        surface_source_t sourceSurface;
        sourceSurface.type = particleSource.at("Type").get<std::string>();
        sourceSurface.count = particleSource.at("Count").get<int>();
        sourceSurface.energy = particleSource.at("Energy").get<double>();
        sourceSurface.baseCoordinates = particleSource.at("BaseCoordinates").get<std::unordered_map<std::string, std::vector<double>>>();

        m_config.particleSourceSurfaces.push_back(sourceSurface);
    }
}

void ConfigParser::_processPicFemParameters(json const &configJson)
{
    m_config.edgeSize = std::stod(configJson.at("EdgeSize").get<std::string>());
    m_config.desiredAccuracy = std::stoi(configJson.at("DesiredAccuracy").get<std::string>());
}

void ConfigParser::_processIterativeSolverParameters(json const &configJson)
{
    // Iterative solver parameters.
    if (configJson.contains("solverName"))
        m_config.solverName = configJson.at("solverName").get<std::string>();
    if (configJson.contains("maxIterations"))
        m_config.maxIterations = std::stoi(configJson.at("maxIterations").get<std::string>());
    if (configJson.contains("convergenceTolerance"))
        m_config.convergenceTolerance = std::stod(configJson.at("convergenceTolerance").get<std::string>());
    if (configJson.contains("verbosity"))
        m_config.verbosity = std::stoi(configJson.at("verbosity").get<std::string>());
    if (configJson.contains("outputFrequency"))
        m_config.outputFrequency = std::stoi(configJson.at("outputFrequency").get<std::string>());
    if (configJson.contains("numBlocks"))
        m_config.numBlocks = std::stoi(configJson.at("numBlocks").get<std::string>());
    if (configJson.contains("blockSize"))
        m_config.blockSize = std::stoi(configJson.at("blockSize").get<std::string>());
    if (configJson.contains("maxRestarts"))
        m_config.maxRestarts = std::stoi(configJson.at("maxRestarts").get<std::string>());
    if (configJson.contains("flexibleGMRES"))
        m_config.flexibleGMRES = configJson.at("flexibleGMRES").get<std::string>() == "true";
    if (configJson.contains("orthogonalization"))
        m_config.orthogonalization = configJson.at("orthogonalization").get<std::string>();
    if (configJson.contains("adaptiveBlockSize"))
        m_config.adaptiveBlockSize = configJson.at("adaptiveBlockSize").get<std::string>() == "true";
    if (configJson.contains("convergenceTestFrequency"))
        m_config.convergenceTestFrequency = std::stoi(configJson.at("convergenceTestFrequency").get<std::string>());
}

void ConfigParser::_processBoundaryConditions(json const &configJson)
{
    if (!configJson.contains("Boundary Conditions"))
        return;

    json boundaryConditionsJson = configJson.at("Boundary Conditions");
    for (auto const &[key, value] : boundaryConditionsJson.items())
    {
        std::vector<size_t> nodes;
        std::string nodesStr(key);
        size_t pos{};
        std::string token;
        while ((pos = nodesStr.find(',')) != std::string::npos)
        {
            token = nodesStr.substr(0ul, pos);
            try
            {
                size_t nodeId{std::stoul(token)};
                nodes.emplace_back(nodeId);
                m_config.nonChangeableNodes.emplace_back(nodeId);
                m_config.nodeValues[nodeId].emplace_back(value.get<double>());
            }
            catch (std::invalid_argument const &e)
            {
                START_THROW_EXCEPTION(UtilsInvalidNodeIDException,
                                      util::stringify("Invalid node ID: ", token, ". Error: ", e.what()));
            }
            catch (std::out_of_range const &e)
            {
                START_THROW_EXCEPTION(UtilsNodeIDOutOfRangeException,
                                      util::stringify("Node ID out of range: ", token, ". Error: ", e.what()));
            }
            nodesStr.erase(0ul, pos + 1ul);
        }
        if (!nodesStr.empty())
        {
            try
            {
                size_t nodeId{std::stoul(nodesStr)};
                nodes.emplace_back(nodeId);
                m_config.nonChangeableNodes.emplace_back(nodeId);
                m_config.nodeValues[nodeId].emplace_back(value.get<double>());
            }
            catch (std::invalid_argument const &e)
            {
                START_THROW_EXCEPTION(UtilsInvalidNodeIDException,
                                      util::stringify("Invalid node ID: ", nodesStr, ". Error: ", e.what()));
            }
            catch (std::out_of_range const &e)
            {
                START_THROW_EXCEPTION(UtilsNodeIDOutOfRangeException,
                                      util::stringify("Node ID out of range: ", nodesStr, ". Error: ", e.what()));
            }
        }

        double val{};
        try
        {
            val = value.get<double>();
        }
        catch (json::type_error const &e)
        {
            START_THROW_EXCEPTION(UtilsInvalidValueForNodeIDsException,
                                  util::stringify("Invalid value for node IDs: ", key, ". Error: ", e.what()));
        }

        m_config.boundaryConditions.emplace_back(nodes, val);
    }

    // Check for duplicate nodes.
    for (auto const &[nodeId, values] : m_config.nodeValues)
    {
        if (values.size() > 1)
        {
            std::cerr << "Node ID " << nodeId << " has multiple values assigned: ";
            for (double val : values)
                std::cerr << val << ' ';
            std::cerr << std::endl;

            START_THROW_EXCEPTION(UtilsDuplicateNodeValuesException,
                                  "Duplicate node values found. Temporary file with boundary conditions has been deleted.");
        }
    }
}

void ConfigParser::getConfigData(std::string_view config)
{
    if (config.empty())
        START_THROW_EXCEPTION(UtilsMissingRequiredParameterException,
                              "Configuration file path is empty.");

    std::ifstream ifs(config.data());
    if (!ifs)
        START_THROW_EXCEPTION(UtilsFailedToOpenConfigurationFileException,
                              util::stringify("Failed to open configuration file: ", config));

    json configJson;
    try
    {
        ifs >> configJson;
    }
    catch (json::exception const &e)
    {
        START_THROW_EXCEPTION(UtilsFailedToParseConfigurationFileException,
                              util::stringify("Error parsing config JSON: ", e.what()));
    }

    try
    {
        _processRequiredParameters(configJson);
        _processSimulationParameters(configJson);
        _processPointParticleSources(configJson);
        _processSurfaceParticleSources(configJson);
        _processPicFemParameters(configJson);
        _processIterativeSolverParameters(configJson);
        _processBoundaryConditions(configJson);

        SUCCESSMSG(util::stringify("Successfully loaded configuration from: ", config));
    }
    catch (json::exception const &e)
    {
        START_THROW_EXCEPTION(UtilsFailedToParseConfigurationFileException,
                              util::stringify("Error parsing config JSON: ", e.what()));
    }
    catch (std::exception const &e)
    {
        START_THROW_EXCEPTION(UtilsGettingConfigDataException,
                              util::stringify("General error: ", e.what()));
    }
    catch (...)
    {
        START_THROW_EXCEPTION(UtilsUnknownException,
                              "Something went wrong when assigning data from the config.");
    }
}

unsigned int ConfigParser::getNumThreads() const
{
    auto num_threads{m_config.num_threads};
    if (num_threads < 1 || num_threads > std::thread::hardware_concurrency())
        START_THROW_EXCEPTION(UtilsNumThreadsOutOfRangeException,
                              util::stringify("The number of threads requested (", num_threads,
                                              ") exceeds the number of hardware threads supported by the system (",
                                              std::thread::hardware_concurrency(),
                                              "). Please run on a system with more resources."));

    unsigned int hardware_threads{std::thread::hardware_concurrency()},
        threshold{static_cast<unsigned int>(hardware_threads * 0.8)};
    if (num_threads > threshold)
    {
        WARNINGMSG(util::stringify("Warning: The number of threads requested (", num_threads,
                                   ") is close to or exceeds 80% of the available hardware threads (", hardware_threads, ").",
                                   " This might cause the system to slow down or become unresponsive because the system also needs resources for its own tasks."));
    }

    return num_threads;
}
