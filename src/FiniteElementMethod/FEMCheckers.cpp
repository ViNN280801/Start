#include "FiniteElementMethod/FEMCheckers.hpp"

void FEMCheckers::checkMeshFile(std::string_view mesh_filename) { util::check_gmsh_mesh_file(mesh_filename); }

void FEMCheckers::checkDesiredAccuracy(short desired_accuracy)
{
    if (desired_accuracy < FEM_LIMITS_NULL_VALUE)
        throw std::underflow_error("Desired calculation accuracy can't be negative");
    if (desired_accuracy == FEM_LIMITS_NULL_VALUE)
        throw std::invalid_argument("Desired calculation accuracy can't be eqaul to 0");
    if (desired_accuracy > FEM_LIMITS_MAX_DESIRED_CALCULATION_ACCURACY)
        throw std::overflow_error(std::format("Desired calculation accuracy can't be greater than {}. Required range: [{}; {}]",
                                              FEM_LIMITS_MAX_DESIRED_CALCULATION_ACCURACY,
                                              FEM_LIMITS_MIN_DESIRED_CALCULATION_ACCURACY, FEM_LIMITS_MAX_DESIRED_CALCULATION_ACCURACY));
}

void FEMCheckers::checkPolynomOrder(short polynom_order)
{
    if (polynom_order < FEM_LIMITS_NULL_VALUE)
        throw std::underflow_error("Desired calculation accuracy can't be negative");
    if (polynom_order == FEM_LIMITS_NULL_VALUE)
        throw std::invalid_argument("Desired calculation accuracy can't be eqaul to 0");
    if (polynom_order > FEM_LIMITS_MAX_POLYNOMIAL_ORDER)
        throw std::overflow_error(std::format("Desired calculation accuracy can't be greater than {}. Required range: [{}; {}]",
                                              FEM_LIMITS_MAX_POLYNOMIAL_ORDER,
                                              FEM_LIMITS_MIN_POLYNOMIAL_ORDER, FEM_LIMITS_MAX_POLYNOMIAL_ORDER));
}

void FEMCheckers::checkCell(CellType cellType)
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
