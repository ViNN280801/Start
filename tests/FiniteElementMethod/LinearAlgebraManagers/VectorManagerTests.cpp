#include <gtest/gtest.h>

#include "FiniteElementMethod/LinearAlgebraManagers/VectorManager.hpp"

// Clean test for constructor
TEST(VectorManagerTest, ConstructorWorks)
{
    VectorManager vectorManager(10);
    EXPECT_EQ(vectorManager.size(), 10);
}

// Dirty test: Try initializing with size 0
TEST(VectorManagerTest, ConstructorWithZeroSize)
{
    VectorManager vectorManager(0);
    EXPECT_EQ(vectorManager.size(), 0);
}

// Clean test for clear method
TEST(VectorManagerTest, ClearMethodWorks)
{
    VectorManager vectorManager(5);
    vectorManager.randomize();
    vectorManager.clear();
    for (GlobalOrdinal i{}; i < static_cast<GlobalOrdinal>(vectorManager.size()); ++i)
        EXPECT_EQ(vectorManager[i], 0.0);
}

// Dirty test: Accessing out of bounds index
TEST(VectorManagerTest, AccessOutOfBoundsIndex)
{
    VectorManager vectorManager(5);
    EXPECT_THROW(vectorManager.at(10), std::out_of_range);
}

// Dirty test: Negative index (only if GlobalOrdinal is signed)
TEST(VectorManagerTest, AccessNegativeIndex)
{
    VectorManager vectorManager(5);
    EXPECT_THROW(vectorManager.at(-1), std::out_of_range);
}

// Clean test for randomize method
TEST(VectorManagerTest, RandomizeMethodWorks)
{
    VectorManager vectorManager(5);
    vectorManager.randomize();
    bool hasNonZero = false;
    for (GlobalOrdinal i{}; i < static_cast<GlobalOrdinal>(vectorManager.size()); ++i)
        if (vectorManager[i] != 0.0)
            hasNonZero = true;
    EXPECT_TRUE(hasNonZero);
}

// Dirty test: Call randomize on zero-sized vector
TEST(VectorManagerTest, RandomizeZeroSizedVector)
{
    VectorManager vectorManager(0);
    EXPECT_NO_THROW(vectorManager.randomize());
}

// Dirty test: Check if setting a value in a cleared vector throws
TEST(VectorManagerTest, SetValueInClearedVector)
{
    VectorManager vectorManager(5);
    vectorManager.clear();
    EXPECT_EQ(vectorManager[0], 0.0);
    vectorManager[0] = 1.0;
    EXPECT_EQ(vectorManager[0], 1.0);
}
