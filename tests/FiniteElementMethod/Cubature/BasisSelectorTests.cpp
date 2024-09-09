#include <gtest/gtest.h>
#include <stdexcept>

#include "FiniteElementMethod/Cubature/BasisSelector.hpp"

extern void supress_output(std::ostream &stream);
extern void restore_output(std::ostream &stream);
extern void execute_without_output(std::function<void()> const &func, std::ostream &stream);

// Test fixture class for BasisSelector
class BasisSelectorTest : public ::testing::Test
{
};

// Clear Test 1: Valid CellType with valid polynomial order
TEST_F(BasisSelectorTest, ValidCellTypeAndPolynomOrder)
{
    auto basis1 = BasisSelector::get<Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>>(CellType::Triangle, 2);
    EXPECT_NE(basis1, nullptr);

    auto basis2 = BasisSelector::get<Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>>(CellType::Tetrahedron, 2);
    EXPECT_NE(basis2, nullptr);
}

// Clear Test 2: Unsupported CellType
TEST_F(BasisSelectorTest, UnsupportedCellType)
{
    try
    {
        auto basis = BasisSelector::get<Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>>(static_cast<CellType>(999), 2);
        FAIL() << "Expected CellSelectorException";
    }
    catch (const CellSelectorException &e)
    {
        SUCCEED();
    }
    catch (const std::exception &e)
    {
        FAIL() << "Expected CellSelectorException, but got: " << typeid(e).name();
    }
}

// Dirty Test 1: Invalid negative polynomial order
TEST_F(BasisSelectorTest, InvalidNegativePolynomOrder)
{
    try
    {
        auto basis = BasisSelector::get<Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>>(CellType::Triangle, -1);
        FAIL() << "Expected std::runtime_error";
    }
    catch (const std::runtime_error &e)
    {
        SUCCEED();
    }
    catch (const std::exception &e)
    {
        FAIL() << "Expected std::runtime_error, but got: " << typeid(e).name();
    }
}

// Dirty Test 2: Zero polynomial order (not allowed)
TEST_F(BasisSelectorTest, ZeroPolynomOrder)
{
    try
    {
        auto basis = BasisSelector::get<Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>>(CellType::Triangle, 0);
        FAIL() << "Expected std::invalid_argument";
    }
    catch (const std::invalid_argument &e)
    {
        SUCCEED();
    }
    catch (const std::exception &e)
    {
        FAIL() << "Expected std::invalid_argument, but got: " << typeid(e).name();
    }
}

// Dirty Test 3: Large polynomial order
TEST_F(BasisSelectorTest, LargePolynomOrder)
{
    try
    {
        auto basis = BasisSelector::get<Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>>(CellType::Triangle, 1000);
        FAIL() << "Expected std::overflow_error due to large polynomial order";
    }
    catch (const std::overflow_error &e)
    {
        SUCCEED();
    }
    catch (...)
    {
        FAIL() << "Expected std::overflow_error due to large polynomial order";
    }
}

// Dirty Test 4: Extreme invalid CellType (underflow)
TEST_F(BasisSelectorTest, ExtremeNegativeCellType)
{
    try
    {
        auto basis = BasisSelector::get<Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>>(static_cast<CellType>(std::numeric_limits<int>::min()), 2);
        FAIL() << "Expected CellSelectorException";
    }
    catch (const CellSelectorException &e)
    {
        SUCCEED();
    }
    catch (const std::exception &e)
    {
        FAIL() << "Expected CellSelectorException, but got: " << typeid(e).name();
    }
}

// Dirty Test 5: Extreme large CellType (overflow)
TEST_F(BasisSelectorTest, ExtremeLargeCellType)
{
    try
    {
        auto basis = BasisSelector::get<Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>>(static_cast<CellType>(std::numeric_limits<int>::max()), 2);
        FAIL() << "Expected CellSelectorException";
    }
    catch (const CellSelectorException &e)
    {
        SUCCEED();
    }
    catch (const std::exception &e)
    {
        FAIL() << "Expected CellSelectorException, but got: " << typeid(e).name();
    }
}

// Dirty Test 6: Unsupported polynomial order for Pyramid (should only support order 1)
TEST_F(BasisSelectorTest, InvalidPolynomOrderForPyramid)
{
    try
    {
        auto basis = BasisSelector::get<Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>>(CellType::Pyramid, 2);
        FAIL() << "Expected std::runtime_error for invalid polynomial order on Pyramid";
    }
    catch (const std::runtime_error &e)
    {
        EXPECT_STREQ(e.what(), "Pyramid cells only support 1st polynomial order.");
    }
    catch (...)
    {
        FAIL() << "Expected std::runtime_error for invalid polynomial order on Pyramid";
    }
}

// Dirty Test 7: Invalid polynomial order for Wedge (should support only 1st and 2nd)
TEST_F(BasisSelectorTest, InvalidPolynomOrderForWedge)
{
    execute_without_output([&]()
                           {
                               auto basis = BasisSelector::get<Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>>(CellType::Wedge, 3);
                               EXPECT_NE(basis, nullptr); // Check if it defaults to valid polynomial order
                           },
                           std::cerr);
}

// Dirty Test 8: Null pointer return check
TEST_F(BasisSelectorTest, NullPointerCheck)
{
    try
    {
        auto basis = BasisSelector::get<Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>>(CellType::Pyramid, -5);
        FAIL() << "Expected std::runtime_error";
    }
    catch (const std::runtime_error &e)
    {
        SUCCEED();
    }
    catch (const std::exception &e)
    {
        FAIL() << "Expected std::runtime_error, but got: " << typeid(e).name();
    }
}

// Dirty Test 9: Valid large polynomial order for valid cell type (Hexahedron)
TEST_F(BasisSelectorTest, ValidLargePolynomOrderForHexahedron)
{
    execute_without_output([&]()
                           {
    auto basis = BasisSelector::get<Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>>(CellType::Hexahedron, 2);
    EXPECT_NE(basis, nullptr); },
                           std::cerr);
}
