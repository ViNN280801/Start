#include <gtest/gtest.h>

#include "FiniteElementMethod/LinearAlgebraManagers/MatrixManager.hpp"

TEST(MatrixManagerTest, ConstructorWithValidEntries)
{
    std::vector<MatrixEntry> entries = {{0, 1, 2.5}, {1, 0, 3.0}, {1, 1, 4.0}};
    EXPECT_NO_THROW(MatrixManager matrixManager(entries));
}

TEST(MatrixManagerTest, ConstructorWithEmptyEntries)
{
    std::vector<MatrixEntry> emptyEntries;
    EXPECT_THROW(MatrixManager matrixManager(emptyEntries), std::invalid_argument);
}

TEST(MatrixManagerTest, ConstructorWithOutOfBoundsEntries)
{
    std::vector<MatrixEntry> outOfBoundsEntries = {{-10001, 1, 2.5}, {1, 0, 3.0}};
    EXPECT_THROW(MatrixManager matrixManager(outOfBoundsEntries), std::out_of_range);
}

TEST(MatrixManagerTest, ConstructorWithNegativeIndices)
{
    std::vector<MatrixEntry> negativeEntries = {{-1, 1, 2.5}, {1, -2, 3.0}};
    EXPECT_THROW(MatrixManager matrixManager(negativeEntries), std::out_of_range);
}

TEST(MatrixManagerTest, ConstructorWithDuplicateEntries)
{
    std::vector<MatrixEntry> duplicateEntries = {{0, 1, 2.5}, {0, 1, 3.5}};
    EXPECT_NO_THROW(MatrixManager matrixManager(duplicateEntries));
}

TEST(MatrixManagerTest, ConstructorWithInvalidCRSMatrix)
{
    std::vector<MatrixEntry> invalidCRS = {{0, 1, 2.5}, {1, -1000, 3.0}};
    EXPECT_THROW(MatrixManager matrixManager(invalidCRS), std::out_of_range);
}

TEST(MatrixManagerTest, AccessValidElement_1)
{
    std::vector<MatrixEntry> entries = {{0, 1, 2.5}, {1, 0, 3.0}, {1, 1, 4.0}};
    MatrixManager matrixManager(entries);
    EXPECT_EQ(matrixManager.at(0, 1), 2.5);
}

TEST(MatrixManagerTest, AccessValidElement_2)
{
    std::vector<MatrixEntry> entries = {{0, 1, 2.5}, {1, 0, 3.0}, {1, 1, 4.0}};
    MatrixManager matrixManager(entries);
    EXPECT_EQ(matrixManager.at(1, 1), 4.0);
}

TEST(MatrixManagerTest, AccessValidElement_3)
{
    std::vector<MatrixEntry> entries = {{2, 2, 1.0}, {0, 0, 2.0}, {1, 1, 3.0}};
    MatrixManager matrixManager(entries);
    EXPECT_EQ(matrixManager.at(2, 2), 1.0);
}

TEST(MatrixManagerTest, AccessValidElement_4)
{
    std::vector<MatrixEntry> entries = {{0, 0, 10.0}, {1, 2, 5.5}, {2, 1, 7.0}};
    MatrixManager matrixManager(entries);
    EXPECT_EQ(matrixManager.at(1, 2), 5.5);
}

TEST(MatrixManagerTest, AccessValidElement_5)
{
    std::vector<MatrixEntry> entries = {{2, 2, 10.0}, {1, 2, 5.5}, {2, 1, 7.0}};
    MatrixManager matrixManager(entries);
    EXPECT_EQ(matrixManager.at(1, 2), 5.5);
}

TEST(MatrixManagerTest, AccessOutOfBoundsRow)
{
    std::vector<MatrixEntry> entries = {{0, 1, 2.5}, {1, 0, 3.0}};
    MatrixManager matrixManager(entries);
    EXPECT_THROW(matrixManager.at(1000, 1), std::out_of_range);
}

TEST(MatrixManagerTest, AccessOutOfBoundsColumn)
{
    std::vector<MatrixEntry> entries = {{0, 1, 2.5}, {1, 0, 3.0}};
    MatrixManager matrixManager(entries);
    EXPECT_THROW(matrixManager.at(1, 1000), std::out_of_range);
}

TEST(MatrixManagerTest, AccessNonExistentElement)
{
    std::vector<MatrixEntry> entries = {{0, 1, 2.5}, {1, 0, 3.0}};
    MatrixManager matrixManager(entries);
    EXPECT_THROW(matrixManager.at(1, 2), std::out_of_range);
}

TEST(MatrixManagerTest, AccessElementAfterInvalidCRS)
{
    std::vector<MatrixEntry> invalidEntries = {{0, -1, 2.5}, {1, 1000, 3.0}};
    EXPECT_THROW(MatrixManager matrixManager(invalidEntries), std::out_of_range);
}

TEST(MatrixManagerTest, ValidMatrixDimensions)
{
    std::vector<MatrixEntry> entries = {{0, 1, 2.5}, {1, 0, 3.0}, {1, 1, 4.0}};
    MatrixManager matrixManager(entries);
    EXPECT_EQ(matrixManager.rows(), 2);
    EXPECT_EQ(matrixManager.cols(), 2);
}

TEST(MatrixManagerTest, EmptyMatrixDimensions)
{
    std::vector<MatrixEntry> emptyEntries;
    EXPECT_THROW(MatrixManager matrixManager(emptyEntries), std::invalid_argument);
}

TEST(MatrixManagerTest, DimensionsWithLargeValues)
{
    std::vector<MatrixEntry> largeEntries = {{100000, -1000, 2.5}, {20000, 2000, 3.0}};
    EXPECT_THROW(MatrixManager matrixManager(largeEntries), std::out_of_range);
}

TEST(MatrixManagerTest, DimensionsAfterFailedCRSGraph)
{
    std::vector<MatrixEntry> invalidEntries = {{0, 1, 2.5}, {1, -10000, 3.0}};
    EXPECT_THROW(MatrixManager matrixManager(invalidEntries), std::out_of_range);
}

TEST(MatrixManagerTest, MatrixIsEmpty)
{
    std::vector<MatrixEntry> emptyEntries;
    EXPECT_THROW(MatrixManager matrixManager(emptyEntries), std::invalid_argument);
}

TEST(MatrixManagerTest, EmptyMatrixUninitialized)
{
    EXPECT_THROW(MatrixManager matrixManager(std::vector<MatrixEntry>{}), std::invalid_argument);
}

TEST(MatrixManagerTest, EmptyMatrixAfterClearing)
{
    std::vector<MatrixEntry> entries = {{0, 1, 2.5}, {1, 0, 3.0}};
    MatrixManager matrixManager(entries);
    // Clear method assumed, manually clear matrix
    EXPECT_FALSE(matrixManager.empty());
}

TEST(MatrixManagerTest, EmptyMatrixWithZeroValues)
{
    std::vector<MatrixEntry> zeroEntries = {{0, 1, 0.0}, {1, 0, 0.0}};
    MatrixManager matrixManager(zeroEntries);
    EXPECT_FALSE(matrixManager.empty());
}

TEST(MatrixManagerTest, EmptyMatrixInvalidCRS)
{
    std::vector<MatrixEntry> invalidEntries = {{0, 1, 2.5}, {-1, 1000, 3.0}};
    EXPECT_THROW(MatrixManager matrixManager(invalidEntries), std::out_of_range);
}
