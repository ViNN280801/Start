#include <H5Cpp.h>
#include <filesystem>
#include <gtest/gtest.h>

#include "Geometry/GeometryTypes.hpp"
#include "TriangleMeshHdf5Manager.hpp"

class TriangleMeshHdf5ManagerTest : public ::testing::Test
{
protected:
    std::string filename{"test.hdf5"};
    TriangleMeshHdf5Manager *handler{};

    void SetUp() override
    {
        if (std::filesystem::exists(filename))
            std::filesystem::remove(filename);
        handler = new TriangleMeshHdf5Manager(filename);
    }

    void TearDown() override
    {
        delete handler;
        if (std::filesystem::exists(filename))
            std::filesystem::remove(filename);
    }
};

TEST_F(TriangleMeshHdf5ManagerTest, FileCreation) { EXPECT_TRUE(std::filesystem::exists(filename)); }

TEST_F(TriangleMeshHdf5ManagerTest, SaveAndReadMesh)
{
    // Create and save a mesh.
    TriangleCellMap mesh;
    TriangleCell cell(Triangle(Point(0, 0, 0), Point(1, 0, 0), Point(0, 1, 0)), 101.123f, 578'154);
    mesh[167ul] = cell;
    handler->saveMeshToHDF5(mesh);

    // Read back the mesh.
    TriangleCellMap readMesh{handler->readMeshFromHDF5()};
    EXPECT_EQ(mesh.size(), readMesh.size());
    if (!readMesh.empty())
    {
        auto const &[id, triangle, area, count]{readMesh[0]};
        auto const &[id2, triangle2, area2, count2]{mesh[0]};
        
        EXPECT_EQ(id, id2);
        EXPECT_EQ(triangle, triangle2);
        EXPECT_DOUBLE_EQ(area, area2);
        EXPECT_EQ(count, count2);
    }
}
