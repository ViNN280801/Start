#include <filesystem>
#include <fstream>
#include <gmsh.h>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>

#include "../include/DataHandling/VolumetricMeshData.hpp"

void createTemporaryBoxMeshFile(std::string const &filename,
                                double meshSize = 10, int dims = 3,
                                double x = 0, double y = 0, double z = 0,
                                double dx = 5, double dy = 5, double dz = 5)
{
    gmsh::model::occ::addBox(x, y, z, dx, dy, dz);
    gmsh::model::occ::synchronize();
    gmsh::option::setNumber("Mesh.MeshSizeMin", meshSize);
    gmsh::option::setNumber("Mesh.MeshSizeMax", meshSize);
    gmsh::model::mesh::generate(dims);
    gmsh::write(filename);
}

void createTemporaryMeshFile(std::string const &filename, std::string const &content)
{
    std::ofstream file(filename);
    if (!file)
        throw std::runtime_error("Failed to create temporary mesh file: " + filename);
    file << content;
    file.close();
}

void removeTemporaryMeshFile(std::string const &filename) { std::filesystem::remove(filename); }

class VolumetricMeshDataTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        gmsh::initialize();
        gmsh::option::setNumber("General.Terminal", 0);

        createTemporaryBoxMeshFile("valid_mesh.msh");
        createTemporaryMeshFile("empty_mesh.msh", "");
        createTemporaryMeshFile("test.txt", "Content\n");
        std::filesystem::create_directory("test_directory");
    }

    void TearDown() override
    {
        gmsh::finalize();

        removeTemporaryMeshFile("valid_mesh.msh");
        removeTemporaryMeshFile("empty_mesh.msh");
        removeTemporaryMeshFile("test.txt");
        std::filesystem::remove_all("test_directory");
    }
};

TEST_F(VolumetricMeshDataTest, SingletonPattern)
{
    auto &instance1{VolumetricMeshData::getInstance("valid_mesh.msh")};
    auto &instance2{VolumetricMeshData::getInstance("valid_mesh.msh")};

    EXPECT_EQ(std::addressof(instance1), std::addressof(instance2)) << "Singleton pattern is violated: multiple instances exist.";
}

TEST_F(VolumetricMeshDataTest, SingletonPatternDifferentFiles)
{
    auto &instance1{VolumetricMeshData::getInstance("valid_mesh.msh")};
    auto &instance2{VolumetricMeshData::getInstance("valid_mesh.msh")};

    EXPECT_EQ(std::addressof(instance1), std::addressof(instance2)) << "Singleton pattern is violated: multiple instances exist for different files.";
}

TEST_F(VolumetricMeshDataTest, InvalidFilePath)
{
    EXPECT_THROW({ VolumetricMeshData::getInstance("invalid_path.msh"); }, std::runtime_error) << "Expected runtime_error for invalid file path.";
}

TEST_F(VolumetricMeshDataTest, DirectoryPath)
{
    EXPECT_THROW({ VolumetricMeshData::getInstance("test_directory"); }, std::runtime_error) << "Expected runtime_error for directory path.";
}

TEST_F(VolumetricMeshDataTest, EmptyFile)
{
    EXPECT_THROW({ VolumetricMeshData::getInstance("empty_mesh.msh"); }, std::runtime_error) << "Expected runtime_error for empty file.";
}

TEST_F(VolumetricMeshDataTest, InvalidExtensionFile)
{
    EXPECT_THROW({ VolumetricMeshData::getInstance("test.txt"); }, std::runtime_error) << "Expected runtime_error for invalid file extension.";
}

TEST_F(VolumetricMeshDataTest, LoadValidMeshData)
{
    auto &instance{VolumetricMeshData::getInstance("valid_mesh.msh")};

    ASSERT_FALSE(instance.empty()) << "Mesh data should not be empty for a valid mesh file.";
    EXPECT_GT(instance.size(), 0) << "Mesh should contain tetrahedra for a valid mesh file.";

    for (auto const &tetra : instance.getMeshComponents())
    {
        EXPECT_FALSE(tetra.electricField.has_value()) << "Electric field should be assigned for each tetrahedron.";
        EXPECT_FALSE(std::all_of(tetra.nodes.begin(), tetra.nodes.end(), [](VolumetricMeshData::NodeData const &node)
                                 { return node.potential.has_value(); }))
            << "Potential should be assigned for each node.";
    }
}

TEST_F(VolumetricMeshDataTest, CalculateTotalVolume)
{
    auto &instance{VolumetricMeshData::getInstance("valid_mesh.msh")};

    double totalVolume{instance.volume()};
    EXPECT_GT(totalVolume, 0.0) << "Total volume should be greater than 0 for a valid mesh.";
}

TEST_F(VolumetricMeshDataTest, GetTetrahedronCenters)
{
    auto &instance{VolumetricMeshData::getInstance("valid_mesh.msh")};

    auto centers{instance.getTetrahedronCenters()};
    EXPECT_EQ(centers.size(), instance.size()) << "Number of centers should match number of tetrahedra.";
}
