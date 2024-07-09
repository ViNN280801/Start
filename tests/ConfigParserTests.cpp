#include <fstream>
#include <gtest/gtest.h>
#include <stdexcept>

#include "../include/Utilities/ConfigParser.hpp"

class ConfigParserTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        CreateFile("valid_config_1.json", R"({
            "Mesh File": "path/to/mesh.msh",
            "Threads": 4,
            "Time Step": 0.01,
            "Simulation Time": 1.0,
            "T": 300.0,
            "P": 1.0,
            "Gas": "He",
            "Model": "VSS",
            "EdgeSize": "1.0",
            "DesiredAccuracy": "2.0",
            "solverName": "GMRES",
            "maxIterations": "1000",
            "convergenceTolerance": "1e-5"
        })");

        CreateFile("valid_config_2.json", R"({
            "Mesh File": "path/to/mesh2.msh",
            "Threads": 8,
            "Time Step": 0.02,
            "Simulation Time": 2.0,
            "T": 350.0,
            "P": 2.0,
            "Gas": "O2",
            "Model": "LTS",
            "EdgeSize": "0.5",
            "DesiredAccuracy": "1.0",
            "solverName": "BiCGSTAB",
            "maxIterations": "2000",
            "convergenceTolerance": "1e-6"
        })");

        CreateFile("invalid_config_1.json", R"({
            "Mesh File": "path/to/mesh.msh",
            "Threads": "four",
            "Time Step": "ten",
            "Simulation Time": "one",
            "T": "three hundred",
            "P": "one",
            "Gas": 2,
            "Model": 1,
            "EdgeSize": "one",
            "DesiredAccuracy": "two",
            "solverName": 3,
            "maxIterations": "thousand",
            "convergenceTolerance": "1e-5"
        })");

        CreateFile("invalid_config_2.json", R"({
            "Mesh File": "path/to/mesh.msh",
            "Threads": -1,
            "Time Step": -0.01,
            "Simulation Time": -1.0,
            "T": -300.0,
            "P": -1.0,
            "Gas": "",
            "Model": "",
            "EdgeSize": -1.0,
            "DesiredAccuracy": -2.0,
            "solverName": "",
            "maxIterations": -1000,
            "convergenceTolerance": -1e-5
        })");

        CreateFile("missing_field_config.json", R"({
            "Mesh File": "path/to/mesh.msh",
            "Threads": 4,
            "Time Step": 0.01,
            "Simulation Time": 1.0,
            "T": 300.0,
            "Gas": "H2",
            "Model": "VSS",
            "EdgeSize": "1.0",
            "DesiredAccuracy": "2.0",
            "solverName": "GMRES",
            "maxIterations": "1000",
            "convergenceTolerance": "1e-5"
        })");

        CreateFile("extra_field_config.json", R"({
            "Mesh File": "path/to/mesh.msh",
            "Threads": 4,
            "Time Step": 0.01,
            "Simulation Time": 1.0,
            "T": 300.0,
            "P": 1.0,
            "Gas": "H2",
            "Model": "VSS",
            "EdgeSize": "1.0",
            "DesiredAccuracy": "2.0",
            "solverName": "GMRES",
            "maxIterations": "1000",
            "convergenceTolerance": "1e-5",
            "Extra Field": "extra_value"
        })");

        CreateFile("empty_config.json", R"({})");

        CreateFile("malformed_json_config.json", R"({
            "Mesh File": "path/to/mesh.msh",
            "Threads": 4,
            "Time Step": 0.01,
            "Simulation Time": 1.0,
            "T": 300.0,
            "P": 1.0,
            "Gas": "H2",
            "Model": "VSS"
            "EdgeSize": "1.0",
            "DesiredAccuracy": "2.0",
            "solverName": "GMRES",
            "maxIterations": "1000",
            "convergenceTolerance": "1e-5"
        )");

        CreateFile("non_existent_file.json", "");

        CreateFile("type_mismatch_config.json", R"({
            "Mesh File": 123,
            "Threads": "four",
            "Time Step": false,
            "Simulation Time": null,
            "T": [],
            "P": {},
            "Gas": 2,
            "Model": true,
            "EdgeSize": "one",
            "DesiredAccuracy": "two",
            "solverName": 3,
            "maxIterations": "thousand",
            "convergenceTolerance": "1e-5"
        })");
    }

    void TearDown() override
    {
        std::remove("valid_config_1.json");
        std::remove("valid_config_2.json");
        std::remove("invalid_config_1.json");
        std::remove("invalid_config_2.json");
        std::remove("missing_field_config.json");
        std::remove("extra_field_config.json");
        std::remove("empty_config.json");
        std::remove("malformed_json_config.json");
        std::remove("non_existent_file.json");
        std::remove("type_mismatch_config.json");
    }

    void CreateFile(const std::string &filename, const std::string &content)
    {
        std::ofstream file(filename);
        file << content;
        file.close();
    }
};

TEST_F(ConfigParserTest, TestValidConfig1)
{
    EXPECT_NO_THROW(ConfigParser parser("valid_config_1.json"));
    ConfigParser parser("valid_config_1.json");

    EXPECT_EQ(parser.getNumThreads(), 4) << "Error in method getNumThreads: expected 4, got " << parser.getNumThreads();
    EXPECT_NEAR(parser.getTimeStep(), 0.01, 1e-6) << "Error in method getTimeStep: expected 0.01, got " << parser.getTimeStep();
    EXPECT_NEAR(parser.getSimulationTime(), 1.0, 1e-6) << "Error in method getSimulationTime: expected 1.0, got " << parser.getSimulationTime();
    EXPECT_NEAR(parser.getTemperature(), 300.0, 1e-6) << "Error in method getTemperature: expected 300.0, got " << parser.getTemperature();
    EXPECT_NEAR(parser.getPressure(), 1.0, 1e-6) << "Error in method getPressure: expected 1.0, got " << parser.getPressure();
    EXPECT_EQ(parser.getGas(), constants::particle_types::He) << "Error in method getGas: expected He, got " << parser.getGas();
    EXPECT_EQ(parser.getScatteringModel(), "VSS") << "Error in method getScatteringModel: expected VSS, got " << parser.getScatteringModel();
    EXPECT_NEAR(parser.getEdgeSize(), 1.0, 1e-6) << "Error in method getEdgeSize: expected 1.0, got " << parser.getEdgeSize();
    EXPECT_NEAR(parser.getDesiredCalculationAccuracy(), 2.0, 1e-6) << "Error in method getDesiredCalculationAccuracy: expected 2.0, got " << parser.getDesiredCalculationAccuracy();
    EXPECT_EQ(parser.getSolverName(), "GMRES") << "Error in method getSolverName: expected GMRES, got " << parser.getSolverName();
    EXPECT_EQ(parser.getMaxIterations(), 1000) << "Error in method getMaxIterations: expected 1000, got " << parser.getMaxIterations();
    EXPECT_NEAR(parser.getConvergenceTolerance(), 1e-5, 1e-6) << "Error in method getConvergenceTolerance: expected 1e-5, got " << parser.getConvergenceTolerance();
}

TEST_F(ConfigParserTest, TestValidConfig2)
{
    ConfigParser parser("valid_config_2.json");

    EXPECT_EQ(parser.getNumThreads(), 8) << "Error in method getNumThreads: expected 8, got " << parser.getNumThreads();
    EXPECT_NEAR(parser.getTimeStep(), 0.02, 1e-6) << "Error in method getTimeStep: expected 0.02, got " << parser.getTimeStep();
    EXPECT_NEAR(parser.getSimulationTime(), 2.0, 1e-6) << "Error in method getSimulationTime: expected 2.0, got " << parser.getSimulationTime();
    EXPECT_NEAR(parser.getTemperature(), 350.0, 1e-6) << "Error in method getTemperature: expected 350.0, got " << parser.getTemperature();
    EXPECT_NEAR(parser.getPressure(), 2.0, 1e-6) << "Error in method getPressure: expected 2.0, got " << parser.getPressure();
    EXPECT_EQ(parser.getGas(), constants::particle_types::O2) << "Error in method getGas: expected O2, got " << parser.getGas();
    EXPECT_EQ(parser.getScatteringModel(), "LTS") << "Error in method getScatteringModel: expected LTS, got " << parser.getScatteringModel();
    EXPECT_NEAR(parser.getEdgeSize(), 0.5, 1e-6) << "Error in method getEdgeSize: expected 0.5, got " << parser.getEdgeSize();
    EXPECT_NEAR(parser.getDesiredCalculationAccuracy(), 1.0, 1e-6) << "Error in method getDesiredCalculationAccuracy: expected 1.0, got " << parser.getDesiredCalculationAccuracy();
    EXPECT_EQ(parser.getSolverName(), "BiCGSTAB") << "Error in method getSolverName: expected BiCGSTAB, got " << parser.getSolverName();
    EXPECT_EQ(parser.getMaxIterations(), 2000) << "Error in method getMaxIterations: expected 2000, got " << parser.getMaxIterations();
    EXPECT_NEAR(parser.getConvergenceTolerance(), 1e-6, 1e-6) << "Error in method getConvergenceTolerance: expected 1e-6, got " << parser.getConvergenceTolerance();
}

TEST_F(ConfigParserTest, TestInvalidConfig1)
{
    EXPECT_THROW({ ConfigParser parser("invalid_config_1.json"); }, std::runtime_error) << "Expected runtime_error exception for invalid configuration.";
}

TEST_F(ConfigParserTest, TestInvalidConfig2)
{
    EXPECT_THROW({ ConfigParser parser("invalid_config_2.json"); }, std::runtime_error) << "Expected runtime_error exception for invalid configuration.";
}

TEST_F(ConfigParserTest, TestMissingFieldConfig)
{
    EXPECT_THROW({ ConfigParser parser("missing_field_config.json"); }, std::runtime_error) << "Expected runtime_error exception for configuration with missing fields.";
}

TEST_F(ConfigParserTest, TestExtraFieldConfig)
{
    EXPECT_NO_THROW({ ConfigParser parser("extra_field_config.json"); });
    ConfigParser parser("extra_field_config.json");
    EXPECT_EQ(parser.getNumThreads(), 4) << "Error in method getNumThreads: expected 4, got " << parser.getNumThreads();
}

TEST_F(ConfigParserTest, TestEmptyConfig)
{
    EXPECT_THROW({ ConfigParser parser("empty_config.json"); }, std::runtime_error) << "Expected runtime_error exception for empty configuration.";
}

TEST_F(ConfigParserTest, TestMalformedJsonConfig)
{
    EXPECT_THROW({ ConfigParser parser("malformed_json_config.json"); }, std::runtime_error) << "Expected runtime_error exception for malformed JSON configuration.";
}

TEST_F(ConfigParserTest, TestNonExistentFile)
{
    EXPECT_THROW({ ConfigParser parser("non_existent_file.json"); }, std::runtime_error) << "Expected runtime_error exception for non-existent configuration file.";
}

TEST_F(ConfigParserTest, TestTypeMismatchConfig)
{
    EXPECT_THROW({ ConfigParser parser("type_mismatch_config.json"); }, std::runtime_error) << "Expected runtime_error exception for type mismatch in configuration.";
}

TEST_F(ConfigParserTest, TestGetMeshFile)
{
    ConfigParser parser("valid_config_1.json");
    EXPECT_EQ(parser.getMeshFilename(), "path/to/mesh.msh") << "Error in method getMeshFilename: expected path/to/mesh.msh, got " << parser.getMeshFilename();
}

TEST_F(ConfigParserTest, TestGetNumThreads)
{
    ConfigParser parser("valid_config_1.json");
    EXPECT_EQ(parser.getNumThreads(), 4) << "Error in method getNumThreads: expected 4, got " << parser.getNumThreads();
}

TEST_F(ConfigParserTest, TestGetTimeStep)
{
    ConfigParser parser("valid_config_1.json");
    EXPECT_NEAR(parser.getTimeStep(), 0.01, 1e-6) << "Error in method getTimeStep: expected 0.01, got " << parser.getTimeStep();
}

TEST_F(ConfigParserTest, TestGetSimulationTime)
{
    ConfigParser parser("valid_config_1.json");
    EXPECT_NEAR(parser.getSimulationTime(), 1.0, 1e-6) << "Error in method getSimulationTime: expected 1.0, got " << parser.getSimulationTime();
}

TEST_F(ConfigParserTest, TestGetTemperature)
{
    ConfigParser parser("valid_config_1.json");
    EXPECT_NEAR(parser.getTemperature(), 300.0, 1e-6) << "Error in method getTemperature: expected 300.0, got " << parser.getTemperature();
}

TEST_F(ConfigParserTest, TestGetPressure)
{
    ConfigParser parser("valid_config_1.json");
    EXPECT_NEAR(parser.getPressure(), 1.0, 1e-6) << "Error in method getPressure: expected 1.0, got " << parser.getPressure();
}

TEST_F(ConfigParserTest, TestGetGas)
{
    ConfigParser parser("valid_config_1.json");
    EXPECT_EQ(parser.getGas(), constants::particle_types::He) << "Error in method getGas: expected He, got " << parser.getGas();
}

TEST_F(ConfigParserTest, TestGetScatteringModel)
{
    ConfigParser parser("valid_config_1.json");
    EXPECT_EQ(parser.getScatteringModel(), "VSS") << "Error in method getScatteringModel: expected VSS, got " << parser.getScatteringModel();
}

TEST_F(ConfigParserTest, TestGetEdgeSize)
{
    ConfigParser parser("valid_config_1.json");
    EXPECT_NEAR(parser.getEdgeSize(), 1.0, 1e-6) << "Error in method getEdgeSize: expected 1.0, got " << parser.getEdgeSize();
}

TEST_F(ConfigParserTest, TestGetDesiredCalculationAccuracy)
{
    ConfigParser parser("valid_config_1.json");
    EXPECT_NEAR(parser.getDesiredCalculationAccuracy(), 2.0, 1e-6) << "Error in method getDesiredCalculationAccuracy: expected 2.0, got " << parser.getDesiredCalculationAccuracy();
}

TEST_F(ConfigParserTest, TestGetSolverName)
{
    ConfigParser parser("valid_config_1.json");
    EXPECT_EQ(parser.getSolverName(), "GMRES") << "Error in method getSolverName: expected GMRES, got " << parser.getSolverName();
}

TEST_F(ConfigParserTest, TestGetMaxIterations)
{
    ConfigParser parser("valid_config_1.json");
    EXPECT_EQ(parser.getMaxIterations(), 1000) << "Error in method getMaxIterations: expected 1000, got " << parser.getMaxIterations();
}

TEST_F(ConfigParserTest, TestGetConvergenceTolerance)
{
    ConfigParser parser("valid_config_1.json");
    EXPECT_NEAR(parser.getConvergenceTolerance(), 1e-5, 1e-6) << "Error in method getConvergenceTolerance: expected 1e-5, got " << parser.getConvergenceTolerance();
}

TEST_F(ConfigParserTest, TestGetVerbosity)
{
    ConfigParser parser("valid_config_1.json");
    EXPECT_EQ(parser.getVerbosity(), 0) << "Error in method getVerbosity: expected 0, got " << parser.getVerbosity();
}

TEST_F(ConfigParserTest, TestGetOutputFrequency)
{
    ConfigParser parser("valid_config_1.json");
    EXPECT_EQ(parser.getOutputFrequency(), 0) << "Error in method getOutputFrequency: expected 0, got " << parser.getOutputFrequency();
}

TEST_F(ConfigParserTest, TestGetNumBlocks)
{
    ConfigParser parser("valid_config_1.json");
    EXPECT_EQ(parser.getNumBlocks(), 0) << "Error in method getNumBlocks: expected 0, got " << parser.getNumBlocks();
}

TEST_F(ConfigParserTest, TestGetBlockSize)
{
    ConfigParser parser("valid_config_1.json");
    EXPECT_EQ(parser.getBlockSize(), 0) << "Error in method getBlockSize: expected 0, got " << parser.getBlockSize();
}

TEST_F(ConfigParserTest, TestGetMaxRestarts)
{
    ConfigParser parser("valid_config_1.json");
    EXPECT_EQ(parser.getMaxRestarts(), 0) << "Error in method getMaxRestarts: expected 0, got " << parser.getMaxRestarts();
}

TEST_F(ConfigParserTest, TestGetFlexibleGMRES)
{
    ConfigParser parser("valid_config_1.json");
    EXPECT_EQ(parser.getFlexibleGMRES(), false) << "Error in method getFlexibleGMRES: expected false, got " << parser.getFlexibleGMRES();
}

TEST_F(ConfigParserTest, TestGetOrthogonalization)
{
    ConfigParser parser("valid_config_1.json");
    EXPECT_EQ(parser.getOrthogonalization(), "") << "Error in method getOrthogonalization: expected empty string, got " << parser.getOrthogonalization();
}

TEST_F(ConfigParserTest, TestGetAdaptiveBlockSize)
{
    ConfigParser parser("valid_config_1.json");
    EXPECT_EQ(parser.getAdaptiveBlockSize(), false) << "Error in method getAdaptiveBlockSize: expected false, got " << parser.getAdaptiveBlockSize();
}

TEST_F(ConfigParserTest, TestGetConvergenceTestFrequency)
{
    ConfigParser parser("valid_config_1.json");
    EXPECT_EQ(parser.getConvergenceTestFrequency(), 0) << "Error in method getConvergenceTestFrequency: expected 0, got " << parser.getConvergenceTestFrequency();
}

TEST_F(ConfigParserTest, TestGetBoundaryConditions)
{
    ConfigParser parser("valid_config_1.json");
    EXPECT_TRUE(parser.getBoundaryConditions().empty()) << "Error in method getBoundaryConditions: expected empty vector, got non-empty vector";
}

TEST_F(ConfigParserTest, TestGetNonChangeableNodes)
{
    ConfigParser parser("valid_config_1.json");
    EXPECT_TRUE(parser.getNonChangeableNodes().empty()) << "Error in method getNonChangeableNodes: expected empty vector, got non-empty vector";
}
