#ifndef FEMCHECKERS_HPP
#define FEMCHECKERS_HPP

#include "Cell/CellSelectorException.hpp"
#include "Cell/CellType.hpp"
#include "FEMLimits.hpp"
#include "Utilities/Utilities.hpp"

/**
 * @class FEMCheckers
 * @brief Provides validation functions for various finite element method (FEM) parameters.
 *
 * The FEMCheckers class offers static methods for validating Gmsh mesh files, calculation accuracy,
 * polynomial orders, and supported cell types. These utility methods ensure the correctness of FEM
 * configurations and throw appropriate exceptions when validation fails.
 */
class FEMCheckers
{
public:
    /**
     * @brief Validates the provided Gmsh mesh file by performing various checks.
     *
     * This function verifies if the provided file exists, is not a directory, has a `.msh` extension,
     * and is not empty. It uses the `util::check_gmsh_mesh_file` method for these validations.
     *
     * The following checks are performed:
     * - The file must exist.
     * - The provided path must not be a directory.
     * - The file must have a `.msh` extension.
     * - The file must not be empty.
     *
     * @param mesh_filename A string view representing the filename or path to the mesh file.
     * @throws std::runtime_error If the file does not exist.
     * @throws std::runtime_error If the provided path is a directory.
     * @throws std::runtime_error If the file extension is not `.msh`.
     * @throws std::runtime_error If the file is empty.
     */
    static void checkMeshFile(std::string_view mesh_filename);

    /**
     * @brief Checks the validity of the desired accuracy parameter.
     *
     * This function ensures that the provided desired accuracy is within the allowable range
     * defined by `FEM_LIMITS_MIN_DESIRED_CALCULATION_ACCURACY` and
     * `FEM_LIMITS_MAX_DESIRED_CALCULATION_ACCURACY`.
     *
     * @param desired_accuracy The desired calculation accuracy to be validated.
     * @throws std::underflow_error If the desired accuracy is negative.
     * @throws std::invalid_argument If the desired accuracy is equal to zero.
     * @throws std::overflow_error If the desired accuracy exceeds the maximum allowable value.
     */
    static void checkDesiredAccuracy(short desired_accuracy);

    /**
     * @brief Checks the validity of the polynomial order parameter.
     *
     * This function ensures that the provided polynomial order is within the allowable range
     * defined by `FEM_LIMITS_MIN_POLYNOMIAL_ORDER` and `FEM_LIMITS_MAX_POLYNOMIAL_ORDER`.
     *
     * @param polynom_order The polynomial order to be validated.
     * @throws std::underflow_error If the polynomial order is negative.
     * @throws std::invalid_argument If the polynomial order is equal to zero.
     * @throws std::overflow_error If the polynomial order exceeds the maximum allowable value.
     */
    static void checkPolynomOrder(short polynom_order);

    /**
     * @brief Validates the given cell type.
     *
     * This method checks if the provided `cellType` is one of the supported types in the `CellType` enum.
     * If the cell type is unsupported, it throws a `CellSelectorException`.
     *
     * Supported cell types:
     * - Triangle
     * - Pentagon
     * - Hexagon
     * - Tetrahedron
     * - Pyramid
     * - Wedge
     * - Hexahedron
     *
     * @param cellType The type of the cell to be validated.
     * @throws CellSelectorException If the provided cell type is not supported.
     */
    static void checkCell(CellType cellType);
};

#endif // !FEMCHECKERS_HPP
