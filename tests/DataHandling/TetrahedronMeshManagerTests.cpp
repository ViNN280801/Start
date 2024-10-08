#include <filesystem>
#include <fstream>
#include <gmsh.h>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>

#include "TetrahedronMeshManager.hpp"

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

class TetrahedronMeshManagerTest : public ::testing::Test
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

TEST_F(TetrahedronMeshManagerTest, InvalidFilePath)
{
    EXPECT_THROW({ TetrahedronMeshManager meshManager("invalid_path.msh"); }, std::runtime_error) << "Expected runtime_error for invalid file path.";
}

TEST_F(TetrahedronMeshManagerTest, DirectoryPath)
{
    EXPECT_THROW({ TetrahedronMeshManager meshManager("test_directory"); }, std::runtime_error) << "Expected runtime_error for directory path.";
}

TEST_F(TetrahedronMeshManagerTest, EmptyFile)
{
    EXPECT_THROW({ TetrahedronMeshManager meshManager("empty_mesh.msh"); }, std::runtime_error) << "Expected runtime_error for empty file.";
}

TEST_F(TetrahedronMeshManagerTest, InvalidExtensionFile)
{
    EXPECT_THROW({ TetrahedronMeshManager meshManager("test.txt"); }, std::runtime_error) << "Expected runtime_error for invalid file extension.";
}

TEST_F(TetrahedronMeshManagerTest, LoadValidMeshData)
{
    TetrahedronMeshManager meshManager("valid_mesh.msh");

    ASSERT_FALSE(meshManager.empty()) << "Mesh data should not be empty for a valid mesh file.";
    EXPECT_GT(meshManager.getNumTetrahedrons(), 0) << "Mesh should contain tetrahedra for a valid mesh file.";

    for (auto const &tetra : meshManager.getMeshComponents())
    {
        EXPECT_FALSE(tetra.electricField.has_value()) << "Electric field should be assigned for each tetrahedron.";
        EXPECT_FALSE(std::all_of(tetra.nodes.begin(), tetra.nodes.end(), [](TetrahedronMeshManager::NodeData const &node)
                                 { return node.potential.has_value(); }))
            << "Potential should be assigned for each node.";
    }
}

TEST_F(TetrahedronMeshManagerTest, CalculateTotalVolume)
{
    TetrahedronMeshManager meshManager("valid_mesh.msh");

    double totalVolume{meshManager.volume()};
    EXPECT_GT(totalVolume, 0.0) << "Total volume should be greater than 0 for a valid mesh.";
}

TEST_F(TetrahedronMeshManagerTest, GetTetrahedronCenters)
{
    TetrahedronMeshManager meshManager("valid_mesh.msh");

    auto centers{meshManager.getTetrahedronCenters()};
    EXPECT_EQ(centers.size(), meshManager.getNumTetrahedrons()) << "Number of centers should match number of tetrahedra.";
}
