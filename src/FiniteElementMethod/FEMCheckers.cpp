#include "FiniteElementMethod/FEMCheckers.hpp"
#include "Utilities/Utilities.hpp"

void FEMCheckers::checkMeshFile(std::string_view mesh_filename) { util::check_gmsh_mesh_file(mesh_filename); }

void FEMCheckers::checkDesiredAccuracy(short desired_accuracy)
{
    if (desired_accuracy < FEM_LIMITS_NULL_VALUE)
        throw std::underflow_error("Desired calculation accuracy can't be negative");
    if (desired_accuracy == FEM_LIMITS_NULL_VALUE)
        throw std::invalid_argument("Desired calculation accuracy can't be 0");
    if (desired_accuracy > FEM_LIMITS_MAX_DESIRED_CALCULATION_ACCURACY)
        throw std::overflow_error(std::format("Desired calculation accuracy can't be greater than {}. Required range: [{}; {}]",
                                              FEM_LIMITS_MAX_DESIRED_CALCULATION_ACCURACY,
                                              FEM_LIMITS_MIN_DESIRED_CALCULATION_ACCURACY, FEM_LIMITS_MAX_DESIRED_CALCULATION_ACCURACY));
}

void FEMCheckers::checkPolynomOrder(short polynom_order)
{
    if (polynom_order < FEM_LIMITS_NULL_VALUE)
        throw std::underflow_error("Polynomial order can't be negative");
    if (polynom_order == FEM_LIMITS_NULL_VALUE)
        throw std::invalid_argument("Polynomial order can't be 0");
    if (polynom_order > FEM_LIMITS_MAX_POLYNOMIAL_ORDER)
        throw std::overflow_error(std::format("Polynomial order can't be greater than {}. Required range: [{}; {}]",
                                              FEM_LIMITS_MAX_POLYNOMIAL_ORDER,
                                              FEM_LIMITS_MIN_POLYNOMIAL_ORDER, FEM_LIMITS_MAX_POLYNOMIAL_ORDER));
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
        THROW_CELL_SELECTOR_EXCEPTION(); // Unsupported cell type, throw exception.
    }
}

void FEMCheckers::checkIndex(GlobalOrdinal index, std::string_view prefix)
{
    // Checking that type is integral
    static_assert(std::is_integral_v<GlobalOrdinal>);

    // If type is integral and signed -> we need to check that the index is not negative
    if constexpr (std::is_signed_v<GlobalOrdinal>)
        if (index < FEM_LIMITS_NULL_VALUE)
        {
            if (prefix != "")
                throw std::out_of_range(util::stringify(prefix, " index cannot be negative"));
            else
                throw std::out_of_range("Index cannot be negative");
        }
}

void FEMCheckers::checkIndex(GlobalOrdinal index, size_t upper_bound, std::string_view prefix)
{
    FEMCheckers::checkIndex(index);

    // Safely compare GlobalOrdinal and size_t
    if constexpr (std::is_signed_v<GlobalOrdinal>)
    {
        // Cast both to int64_t for signed comparison
        if (static_cast<int64_t>(index) >= static_cast<int64_t>(upper_bound))
        {
            if (prefix != "")
                throw std::out_of_range(util::stringify(prefix, " index cannot be bigger than ", upper_bound));
            else
                throw std::out_of_range(util::stringify("Index cannot be bigger than ", upper_bound));
        }
    }
    else
    {
        // For unsigned GlobalOrdinal, simple comparison
        if (static_cast<size_t>(index) >= upper_bound)
        {
            if (prefix != "")
                throw std::out_of_range(util::stringify(prefix, " index cannot be bigger than ", upper_bound));
            else
                throw std::out_of_range(util::stringify("Index cannot be bigger than ", upper_bound));
        }
    }
}