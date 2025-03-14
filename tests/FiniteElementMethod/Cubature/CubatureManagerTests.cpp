#include <gtest/gtest.h>
#include <stdexcept>

#include "FiniteElementMethod/Cell/CellSelectorException.hpp"
#include "FiniteElementMethod/Cubature/CubatureManager.hpp"
#include "FiniteElementMethod/Utils/FEMLimits.hpp"

extern void supress_output(std::ostream &stream);
extern void restore_output(std::ostream &stream);
extern void execute_without_output(std::function<void()> const &func, std::ostream &stream);

// Test fixture class for CubatureManager
class CubatureManagerTest : public ::testing::Test
{
protected:
    // Helper function to check the number of cubature points for a given accuracy
    void checkCubaturePoints(CellType cellType, short desired_accuracy, unsigned short expected_points)
    {
        CubatureManager cubatureManager(cellType, desired_accuracy, FEM_LIMITS_DEFAULT_POLYNOMIAL_ORDER);
        EXPECT_EQ(cubatureManager.getCountCubaturePoints(), expected_points);
    }
};

// Clear Test 1: Valid CellType with valid desired accuracy
TEST_F(CubatureManagerTest, ValidCellTypeAndDesiredAccuracy)
{
    // Using Tetrahedron as CellType
    CubatureManager cubatureManager(CellType::Tetrahedron, 2, 1);
    EXPECT_GT(cubatureManager.getCountCubaturePoints(), 0);
}

// Clear Test 2: Check cubature points match the table for Tetrahedron
TEST_F(CubatureManagerTest, CubaturePointsMatchTable_Tetrahedron)
{
    std::map<short, unsigned short> accuracy_to_points = {
        {1, 1},
        {2, 4},
        {3, 5},
        {4, 11},
        {5, 14},
        {6, 24},
        {7, 31},
        {8, 43},
        {9, 126},
        {10, 126},
        {11, 126},
        {12, 210},
        {13, 210},
        {14, 330},
        {15, 330},
        {16, 495},
        {17, 495},
        {18, 715},
        {19, 715},
        {20, 1001},
    };

    for (auto const &[accuracy, expected_points] : accuracy_to_points)
    {
        SCOPED_TRACE(std::string("Desired accuracy: ") + std::to_string(accuracy));
        checkCubaturePoints(CellType::Tetrahedron, accuracy, expected_points);
    }
}

// Dirty Test 1: Unsupported CellType
TEST_F(CubatureManagerTest, UnsupportedCellType)
{
    try
    {
        CubatureManager cubatureManager(static_cast<CellType>(999), 2, 1);
        FAIL() << "Expected CellSelectorExceptionion";
    }
    catch (CellSelectorException const &)
    {
        SUCCEED();
    }
    catch (const std::exception &e)
    {
        FAIL() << "Expected CellSelectorExceptionion, but got: " << typeid(e).name() << " with message: " << e.what();
    }
}

// Dirty Test 2: Negative desired accuracy
TEST_F(CubatureManagerTest, NegativeDesiredAccuracy)
{
    try
    {
        CubatureManager cubatureManager(CellType::Tetrahedron, -1, 1);
        FAIL() << "Expected std::underflow_error";
    }
    catch (const std::underflow_error &e)
    {
        EXPECT_STREQ(e.what(), "Desired calculation accuracy can't be negative");
    }
    catch (...)
    {
        FAIL() << "Expected std::underflow_error due to negative desired accuracy";
    }
}

// Dirty Test 3: Zero desired accuracy
TEST_F(CubatureManagerTest, ZeroDesiredAccuracy)
{
    try
    {
        CubatureManager cubatureManager(CellType::Tetrahedron, 0, 1);
        FAIL() << "Expected std::invalid_argument";
    }
    catch (const std::invalid_argument &e)
    {
        EXPECT_STREQ(e.what(), "Desired calculation accuracy can't be 0");
    }
    catch (...)
    {
        FAIL() << "Expected std::invalid_argument due to zero desired accuracy";
    }
}

// Dirty Test 4: Desired accuracy exceeding maximum limit
TEST_F(CubatureManagerTest, DesiredAccuracyExceedsMaximum)
{
    try
    {
        CubatureManager cubatureManager(CellType::Tetrahedron, FEM_LIMITS_MAX_DESIRED_CALCULATION_ACCURACY + 1, 1);
        FAIL() << "Expected std::overflow_error";
    }
    catch (const std::overflow_error &e)
    {
        EXPECT_STREQ(e.what(),
                     util::stringify("Desired calculation accuracy can't be greater than ", FEM_LIMITS_MAX_DESIRED_CALCULATION_ACCURACY,
                                     ". Required range: [", FEM_LIMITS_MIN_DESIRED_CALCULATION_ACCURACY, "; ",
                                     FEM_LIMITS_MAX_DESIRED_CALCULATION_ACCURACY, "]")
                         .c_str());
    }
    catch (...)
    {
        FAIL() << "Expected std::overflow_error due to desired accuracy exceeding maximum limit";
    }
}

// Dirty Test 5: Extreme invalid CellType (underflow)
TEST_F(CubatureManagerTest, ExtremeNegativeCellType)
{
    try
    {
        CubatureManager cubatureManager(static_cast<CellType>(std::numeric_limits<int>::min()), 2, 1);
        FAIL() << "Expected CellSelectorExceptionion";
    }
    catch (CellSelectorException const &)
    {
        SUCCEED();
    }
    catch (const std::exception &e)
    {
        FAIL() << "Expected CellSelectorExceptionion, but got: " << typeid(e).name() << " with message: " << e.what();
    }
}

// Dirty Test 6: Extreme large CellType (overflow)
TEST_F(CubatureManagerTest, ExtremeLargeCellType)
{
    try
    {
        CubatureManager cubatureManager(static_cast<CellType>(std::numeric_limits<int>::max()), 2, 1);
        FAIL() << "Expected CellSelectorExceptionion";
    }
    catch (CellSelectorException const &)
    {
        SUCCEED();
    }
    catch (const std::exception &e)
    {
        FAIL() << "Expected CellSelectorExceptionion, but got: " << typeid(e).name() << " with message: " << e.what();
    }
}

// Dirty Test 7: Negative polynomial order
TEST_F(CubatureManagerTest, NegativePolynomOrder)
{
    try
    {
        CubatureManager cubatureManager(CellType::Tetrahedron, 2, -1);
        FAIL() << "Expected std::underflow_error";
    }
    catch (const std::underflow_error &e)
    {
        EXPECT_STREQ(e.what(), "Polynomial order can't be negative");
    }
    catch (...)
    {
        FAIL() << "Expected std::underflow_error due to negative polynomial order";
    }
}

// Dirty Test 8: Zero polynomial order
TEST_F(CubatureManagerTest, ZeroPolynomOrder)
{
    try
    {
        CubatureManager cubatureManager(CellType::Tetrahedron, 2, 0);
        FAIL() << "Expected std::invalid_argument";
    }
    catch (const std::invalid_argument &e)
    {
        EXPECT_STREQ(e.what(), "Polynomial order can't be 0");
    }
    catch (...)
    {
        FAIL() << "Expected std::invalid_argument due to zero polynomial order";
    }
}

// Dirty Test 9: Polynomial order exceeding maximum limit
TEST_F(CubatureManagerTest, PolynomOrderExceedsMaximum)
{
    try
    {
        CubatureManager cubatureManager(CellType::Tetrahedron, 2, FEM_LIMITS_MAX_POLYNOMIAL_ORDER + 1);
        FAIL() << "Expected std::overflow_error";
    }
    catch (const std::overflow_error &e)
    {
        EXPECT_STREQ(e.what(),
                     util::stringify("Polynomial order can't be greater than ", FEM_LIMITS_MAX_POLYNOMIAL_ORDER, ". Required range: [",
                                     FEM_LIMITS_MIN_POLYNOMIAL_ORDER, "; ",
                                     FEM_LIMITS_MAX_POLYNOMIAL_ORDER, "]")
                         .c_str());
    }
    catch (...)
    {
        FAIL() << "Expected std::overflow_error due to polynomial order exceeding maximum limit";
    }
}

// Clear Test 4: Valid maximum polynomial order
TEST_F(CubatureManagerTest, ValidMaximumPolynomOrder)
{
    execute_without_output([&]()
                           {
    CubatureManager cubatureManager(CellType::Tetrahedron, 2, FEM_LIMITS_MAX_POLYNOMIAL_ORDER);
    EXPECT_GT(cubatureManager.getCountCubaturePoints(), 0); },
                           std::cerr);
}

// Clear Test 5: Valid maximum desired accuracy
TEST_F(CubatureManagerTest, ValidMaximumDesiredAccuracy)
{
    CubatureManager cubatureManager(CellType::Tetrahedron, FEM_LIMITS_MAX_DESIRED_CALCULATION_ACCURACY, 1);
    EXPECT_GT(cubatureManager.getCountCubaturePoints(), 0);
}

// Dirty Test 10: Desired accuracy below minimum limit
TEST_F(CubatureManagerTest, DesiredAccuracyBelowMinimum)
{
    try
    {
        CubatureManager cubatureManager(CellType::Tetrahedron, -1, 1);
        FAIL() << "Expected std::underflow_error";
    }
    catch (const std::underflow_error &e)
    {
        SUCCEED();
    }
    catch (...)
    {
        FAIL() << "Expected std::underflow_error due to desired accuracy below minimum limit";
    }
}

// Dirty Test 11: Polynomial order below minimum limit
TEST_F(CubatureManagerTest, PolynomOrderBelowMinimum)
{
    try
    {
        CubatureManager cubatureManager(CellType::Tetrahedron, 2, -1);
        FAIL() << "Expected std::underflow_error";
    }
    catch (const std::underflow_error &e)
    {
        SUCCEED();
    }
    catch (...)
    {
        FAIL() << "Expected std::underflow_error due to polynomial order below minimum limit";
    }
}
