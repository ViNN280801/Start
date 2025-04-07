#include "FiniteElementMethod/Utils/FEMCheckers.hpp"
#include "FiniteElementMethod/FEMExceptions.hpp"
#include "Utilities/Utilities.hpp"

void FEMCheckers::checkDesiredAccuracy(short desired_accuracy)
{
    if (desired_accuracy < FEM_LIMITS_NULL_VALUE)
    {
        START_THROW_EXCEPTION(FEMCheckersUnderflowDesiredAccuracyException,
                              "Desired calculation accuracy can't be negative");
    }
    if (desired_accuracy == FEM_LIMITS_NULL_VALUE)
    {
        START_THROW_EXCEPTION(FEMCheckersUnsupportedDesiredAccuracyException,
                              "Desired calculation accuracy can't be 0");
    }
    if (desired_accuracy > FEM_LIMITS_MAX_DESIRED_CALCULATION_ACCURACY)
    {
        START_THROW_EXCEPTION(FEMCheckersOverflowDesiredAccuracyException,
                              "Desired calculation accuracy can't be greater than " +
                                  std::to_string(FEM_LIMITS_MAX_DESIRED_CALCULATION_ACCURACY) +
                                  ". Required range: [" +
                                  std::to_string(FEM_LIMITS_MIN_DESIRED_CALCULATION_ACCURACY) + "; " +
                                  std::to_string(FEM_LIMITS_MAX_DESIRED_CALCULATION_ACCURACY) + "]");
    }
}

void FEMCheckers::checkPolynomOrder(short polynom_order)
{
    if (polynom_order < FEM_LIMITS_NULL_VALUE)
    {
        START_THROW_EXCEPTION(FEMCheckersUnderflowPolynomOrderException,
                              util::stringify("Polynomial order can't be negative, passed: ", polynom_order));
    }
    if (polynom_order == FEM_LIMITS_NULL_VALUE)
    {
        START_THROW_EXCEPTION(FEMCheckersUnsupportedPolynomOrderException,
                              util::stringify("Polynomial order can't be 0, passed: ", polynom_order));
    }
    if (polynom_order > FEM_LIMITS_MAX_POLYNOMIAL_ORDER)
    {
        START_THROW_EXCEPTION(FEMCheckersOverflowPolynomOrderException,
                              util::stringify("Polynomial order can't be greater than ",
                                              FEM_LIMITS_MAX_POLYNOMIAL_ORDER, ". Required range: [",
                                              FEM_LIMITS_MIN_POLYNOMIAL_ORDER, "; ",
                                              FEM_LIMITS_MAX_POLYNOMIAL_ORDER, "]"));
    }
}

void FEMCheckers::checkCellType(CellType cellType)
{
    switch (cellType)
    {
    case CellType::Triangle:
    case CellType::Pentagon:
    case CellType::Hexagon:
    case CellType::Tetrahedron:
    case CellType::Pyramid:
    case CellType::Wedge:
    case CellType::Hexahedron:
        break; // Supported cell types, no action needed.
    default:
        START_THROW_EXCEPTION(FEMCheckersUnsupportedCellTypeException,
                              "Unsupported cell type");
    }
}

void FEMCheckers::checkIndex(GlobalOrdinal index, std::string_view prefix)
{
    // Checking that type is integral
    static_assert(std::is_integral_v<GlobalOrdinal>);

    // If type is integral and signed -> we need to check that the index is not negative
    if constexpr (std::is_signed_v<GlobalOrdinal>)
    {
        if (index < FEM_LIMITS_NULL_VALUE)
        {
            if (prefix != "")
            {
                START_THROW_EXCEPTION(FEMCheckersIndexOutOfRangeException,
                                      util::stringify(prefix, " index cannot be negative"));
            }
            else
            {
                START_THROW_EXCEPTION(FEMCheckersIndexOutOfRangeException,
                                      "Index cannot be negative");
            }
        }
    }
    else
    {
        if (index < FEM_LIMITS_NULL_VALUE)
        {
            START_THROW_EXCEPTION(FEMCheckersIndexOutOfRangeException, "Index cannot be negative");
        }
    }
}

void FEMCheckers::checkIndex(GlobalOrdinal index, size_t upper_bound, std::string_view prefix)
{
    FEMCheckers::checkIndex(index);

    // Safely compare GlobalOrdinal and size_t
    if constexpr (std::is_signed_v<GlobalOrdinal>)
    {
        // Cast both to int64_t for signed comparison
        if (static_cast<int64_t>(index) > static_cast<int64_t>(upper_bound))
        {
            if (prefix != "")
            {
                START_THROW_EXCEPTION(FEMCheckersIndexOutOfRangeException,
                                      util::stringify(prefix, " index cannot be bigger than ", upper_bound));
            }
            else
            {
                START_THROW_EXCEPTION(FEMCheckersIndexOutOfRangeException,
                                      util::stringify("Index cannot be bigger than ", upper_bound));
            }
        }
    }
    else
    {
        // For unsigned GlobalOrdinal, simple comparison
        if (static_cast<size_t>(index) > upper_bound)
        {
            if (prefix != "")
            {
                START_THROW_EXCEPTION(FEMCheckersIndexOutOfRangeException,
                                      util::stringify(prefix, " index cannot be bigger than ",
                                                      upper_bound));
            }
            else
            {
                START_THROW_EXCEPTION(FEMCheckersIndexOutOfRangeException,
                                      util::stringify("Index cannot be bigger than ",
                                                      upper_bound));
            }
        }
    }
}
