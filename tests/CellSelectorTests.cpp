#include <gtest/gtest.h>
#include <stdexcept>

#include "../include/FiniteElementMethod/Cell/CellSelector.hpp"

class CellSelectorTest : public ::testing::Test
{
};

/// Clear Test 1: Valid Cell Type Selection
TEST_F(CellSelectorTest, ValidCellTypeSelection)
{
    EXPECT_NO_THROW({
        shards::CellTopology topo = CellSelector::getCellType(CellType::Triangle);
        EXPECT_EQ(topo.getKey(), shards::getCellTopologyData<shards::Triangle<3>>()->key);
    });

    EXPECT_NO_THROW({
        shards::CellTopology topo = CellSelector::getCellType(CellType::Pentagon);
        EXPECT_EQ(topo.getKey(), shards::getCellTopologyData<shards::Pentagon<5>>()->key);
    });

    EXPECT_NO_THROW({
        shards::CellTopology topo = CellSelector::getCellType(CellType::Hexagon);
        EXPECT_EQ(topo.getKey(), shards::getCellTopologyData<shards::Hexagon<6>>()->key);
    });

    EXPECT_NO_THROW({
        shards::CellTopology topo = CellSelector::getCellType(CellType::Tetrahedron);
        EXPECT_EQ(topo.getKey(), shards::getCellTopologyData<shards::Tetrahedron<4>>()->key);
    });

    EXPECT_NO_THROW({
        shards::CellTopology topo = CellSelector::getCellType(CellType::Pyramid);
        EXPECT_EQ(topo.getKey(), shards::getCellTopologyData<shards::Pyramid<5>>()->key);
    });

    EXPECT_NO_THROW({
        shards::CellTopology topo = CellSelector::getCellType(CellType::Wedge);
        EXPECT_EQ(topo.getKey(), shards::getCellTopologyData<shards::Wedge<6>>()->key);
    });

    EXPECT_NO_THROW({
        shards::CellTopology topo = CellSelector::getCellType(CellType::Hexahedron);
        EXPECT_EQ(topo.getKey(), shards::getCellTopologyData<shards::Hexahedron<8>>()->key);
    });
}

/// Clear Test 2: Unsupported Cell Type
TEST_F(CellSelectorTest, UnsupportedCellType)
{
    try
    {
        CellSelector::getCellType(static_cast<CellType>(-1));
        FAIL() << "Expected CellSelectorException";
    }
    catch (CellSelectorException const &e)
    {
        SUCCEED();
    }
    catch (std::exception const &e)
    {
        FAIL() << "Expected CellSelectorException, but got: " << typeid(e).name();
    }
}

/// Dirty Test 1: Invalid Numeric Value
TEST_F(CellSelectorTest, InvalidNumericValue)
{
    try
    {
        CellSelector::getCellType(static_cast<CellType>(1000)); // Out of bounds enum value
        FAIL() << "Expected CellSelectorException";
    }
    catch (CellSelectorException const &e)
    {
        SUCCEED();
    }
    catch (std::exception const &e)
    {
        FAIL() << "Expected CellSelectorException, but got: " << typeid(e).name();
    }
}

/// Dirty Test 2: Enum Underflow
TEST_F(CellSelectorTest, EnumUnderflow)
{
    try
    {
        CellSelector::getCellType(static_cast<CellType>(-100)); // Underflowing enum value
        FAIL() << "Expected CellSelectorException";
    }
    catch (CellSelectorException const &e)
    {
        SUCCEED();
    }
    catch (std::exception const &e)
    {
        FAIL() << "Expected CellSelectorException, but got: " << typeid(e).name();
    }
}

/// Dirty Test 3: Enum Overflow
TEST_F(CellSelectorTest, EnumOverflow)
{
    try
    {
        CellSelector::getCellType(static_cast<CellType>(std::numeric_limits<int>::max())); // Overflow enum value
        FAIL() << "Expected CellSelectorException";
    }
    catch (CellSelectorException const &e)
    {
        SUCCEED();
    }
    catch (std::exception const &e)
    {
        FAIL() << "Expected CellSelectorException, but got: " << typeid(e).name();
    }
}

/// Dirty Test 4: Extreme Negative Enum Value
TEST_F(CellSelectorTest, ExtremeNegativeEnumValue)
{
    try
    {
        CellSelector::getCellType(static_cast<CellType>(std::numeric_limits<int>::min())); // Extremely negative value
        FAIL() << "Expected CellSelectorException";
    }
    catch (CellSelectorException const &e)
    {
        SUCCEED();
    }
    catch (std::exception const &e)
    {
        FAIL() << "Expected CellSelectorException, but got: " << typeid(e).name();
    }
}

/// Dirty Test 5: Custom Enum Value
TEST_F(CellSelectorTest, CustomEnumValue)
{
    try
    {
        CellType customCellType = static_cast<CellType>(999); // Custom, unsupported enum value
        CellSelector::getCellType(customCellType);
        FAIL() << "Expected CellSelectorException";
    }
    catch (CellSelectorException const &e)
    {
        SUCCEED();
    }
    catch (std::exception const &e)
    {
        FAIL() << "Expected CellSelectorException, but got: " << typeid(e).name();
    }
}
