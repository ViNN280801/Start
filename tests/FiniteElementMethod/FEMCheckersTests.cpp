#include <gmsh.h>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

#include "FiniteElementMethod/Cell/CellSelectorException.hpp"
#include "FiniteElementMethod/FEMCheckers.hpp"
#include "Utilities/Utilities.hpp"

// Test fixture for FEMCheckers
class FEMCheckersTest : public ::testing::Test
{
protected:
    static void SetUpTestSuite() { gmsh::initialize(); }
    static void TearDownTestSuite() { gmsh::finalize(); }

    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(FEMCheckersTest, ValidMeshFile)
{
    // Step 1: Copy the original mesh file to a new one without the .gtest suffix.
    std::string originalFile = "../meshes/TriangleTestMesh.msh.gtest";
    std::string copiedFile = "../meshes/TriangleTestMesh.msh";

    // Open the original file and the destination file.
    std::ifstream src(originalFile, std::ios::binary);
    std::ofstream dst(copiedFile, std::ios::binary);

    // Check if the original file exists and copy it
    ASSERT_TRUE(src.is_open()) << "Original mesh file does not exist: " << originalFile;
    dst << src.rdbuf();

    // Close the file streams
    src.close();
    dst.close();

    // Step 2: Check the copied file using FEMCheckers.
    EXPECT_NO_THROW(FEMCheckers::checkMeshFile(copiedFile));

    // Step 3: Delete the copied file
    EXPECT_EQ(std::remove(copiedFile.c_str()), 0) << "Failed to delete the copied mesh file: " << copiedFile;
}

TEST_F(FEMCheckersTest, InvalidMeshFile)
{
    EXPECT_THROW(FEMCheckers::checkMeshFile("invalid_mesh.msh"), std::runtime_error);
}

TEST_F(FEMCheckersTest, DesiredAccuracyNegative)
{
    EXPECT_THROW(FEMCheckers::checkDesiredAccuracy(-1), std::underflow_error);
}

TEST_F(FEMCheckersTest, DesiredAccuracyZero)
{
    EXPECT_THROW(FEMCheckers::checkDesiredAccuracy(0), std::invalid_argument);
}

TEST_F(FEMCheckersTest, DesiredAccuracyTooHigh)
{
    EXPECT_THROW(FEMCheckers::checkDesiredAccuracy(FEM_LIMITS_MAX_DESIRED_CALCULATION_ACCURACY + 1), std::overflow_error);
}

TEST_F(FEMCheckersTest, DesiredAccuracyValid)
{
    EXPECT_NO_THROW(FEMCheckers::checkDesiredAccuracy(1));
}

TEST_F(FEMCheckersTest, PolynomOrderNegative)
{
    EXPECT_THROW(FEMCheckers::checkPolynomOrder(-1), std::underflow_error);
}

TEST_F(FEMCheckersTest, PolynomOrderZero)
{
    EXPECT_THROW(FEMCheckers::checkPolynomOrder(0), std::invalid_argument);
}

TEST_F(FEMCheckersTest, PolynomOrderTooHigh)
{
    EXPECT_THROW(FEMCheckers::checkPolynomOrder(FEM_LIMITS_MAX_POLYNOMIAL_ORDER + 1), std::overflow_error);
}

TEST_F(FEMCheckersTest, PolynomOrderValid)
{
    EXPECT_NO_THROW(FEMCheckers::checkPolynomOrder(1));
}

TEST_F(FEMCheckersTest, ValidCellType)
{
    EXPECT_NO_THROW(FEMCheckers::checkCellType(CellType::Tetrahedron));
}

TEST_F(FEMCheckersTest, InvalidCellType)
{
    EXPECT_THROW(FEMCheckers::checkCellType(static_cast<CellType>(999)), CellSelectorException);
}

TEST_F(FEMCheckersTest, IndexNegativeSigned)
{
    GlobalOrdinal signedIndex = -1;
    EXPECT_THROW(FEMCheckers::checkIndex(signedIndex), std::out_of_range);
}

TEST_F(FEMCheckersTest, IndexValidUnsigned)
{
    GlobalOrdinal unsignedIndex = 5;
    EXPECT_NO_THROW(FEMCheckers::checkIndex(unsignedIndex));
}

TEST_F(FEMCheckersTest, IndexTooHighUnsigned)
{
    GlobalOrdinal unsignedIndex = 1000;
    size_t upper_bound = 999;
    EXPECT_THROW(FEMCheckers::checkIndex(unsignedIndex, upper_bound), std::out_of_range);
}

TEST_F(FEMCheckersTest, IndexValidWithinUpperBound)
{
    GlobalOrdinal validIndex = 500;
    size_t upper_bound = 999;
    EXPECT_NO_THROW(FEMCheckers::checkIndex(validIndex, upper_bound));
}

TEST_F(FEMCheckersTest, IndexNegativeWithUpperBound)
{
    GlobalOrdinal signedIndex = -1;
    size_t upper_bound = 100;
    EXPECT_THROW(FEMCheckers::checkIndex(signedIndex, upper_bound), std::out_of_range);
}

TEST_F(FEMCheckersTest, MaxGlobalOrdinal)
{
    GlobalOrdinal maxIndex = 7;
    size_t upper_bound = 6;
    EXPECT_THROW(FEMCheckers::checkIndex(maxIndex, upper_bound), std::out_of_range);
}

TEST_F(FEMCheckersTest, MinGlobalOrdinal)
{
    GlobalOrdinal minIndex = std::numeric_limits<GlobalOrdinal>::min();
    if constexpr (std::is_signed_v<GlobalOrdinal>)
    {
        EXPECT_THROW(FEMCheckers::checkIndex(minIndex), std::out_of_range);
    }
    else
    {
        EXPECT_NO_THROW(FEMCheckers::checkIndex(minIndex));
    }
}

TEST_F(FEMCheckersTest, UpperBoundEdgeCase)
{
    GlobalOrdinal index = 999;
    size_t upper_bound = 999;
    EXPECT_THROW(FEMCheckers::checkIndex(index, upper_bound), std::out_of_range);
}
