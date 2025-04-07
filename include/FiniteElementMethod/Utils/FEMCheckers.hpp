#ifndef FEM_CHECKERS_HPP
#define FEM_CHECKERS_HPP

#include "FiniteElementMethod/FEMExceptions.hpp"
#include "FiniteElementMethod/Cell/CellType.hpp"
#include "FiniteElementMethod/Utils/FEMLimits.hpp"
#include "FiniteElementMethod/FEMTypes.hpp"
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
    static void checkCellType(CellType cellType);

    /**
     * @brief Checks if the provided index is valid and not negative (for signed types).
     *
     * This method performs a check on the given `index` to ensure that:
     * - It is a valid integral type.
     * - If `GlobalOrdinal` is a signed type, the index is not negative.
     *
     * @tparam GlobalOrdinal The type of the index, which must be an integral type.
     * @param index The index to be checked.
     * @param prefix A string prefix that is added to the error message, which helps in identifying where the error occurred.
     *
     * @throw std::invalid_argument If the index is negative for signed types.
     *
     * @note This function uses `static_assert` to ensure that `GlobalOrdinal` is an integral type. It only checks for negative
     * values if the index is of a signed type.
     */
    static void checkIndex(GlobalOrdinal index, std::string_view prefix = "");

    /**
     * @brief Checks if the provided index is valid and within the specified upper bound.
     *
     * This method performs two checks:
     * - The index is not negative if `GlobalOrdinal` is a signed type.
     * - The index does not exceed the specified `upper_bound`.
     *
     * @tparam GlobalOrdinal The type of the index, which must be an integral type.
     * @param index The index to be checked.
     * @param upper_bound The maximum allowed value for the index.
     * @param prefix A string prefix that is added to the error message, which helps in identifying where the error occurred.
     *
     * @throw std::invalid_argument If the index is negative or exceeds the `upper_bound`.
     *
     * @note The function safely handles comparisons between signed and unsigned types by casting both to a common type when necessary.
     */
    static void checkIndex(GlobalOrdinal index, size_t upper_bound, std::string_view prefix = "");
};

#endif // !FEM_CHECKERS_HPP
