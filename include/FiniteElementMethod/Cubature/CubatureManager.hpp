#ifndef CUBATUREMANAGER_HPP
#define CUBATUREMANAGER_HPP

#include "FiniteElementMethod/Cell/CellType.hpp"
#include "FiniteElementMethod/FEMTypes.hpp"

/**
 * @class CubatureManager
 * @brief Manages cubature points and weights for finite element integration.
 *
 * The `CubatureManager` class is responsible for initializing and managing cubature (integration) points
 * and weights for different cell types in finite element analysis. It uses a specified polynomial order
 * and accuracy to generate the appropriate cubature data.
 *
 * Table example for TETRAHEDRON cell type:
 *
 * | FEM accuracy | Count of cubature points |
 * | :----------: | :----------------------: |
 * |      1       |            1             |
 * |      2       |            4             |
 * |      3       |            5             |
 * |      4       |            11            |
 * |      5       |            14            |
 * |      6       |            24            |
 * |      7       |            31            |
 * |      8       |            43            |
 * |      9       |           126            |
 * |      10      |           126            |
 * |      11      |           126            |
 * |      12      |           210            |
 * |      13      |           210            |
 * |      14      |           330            |
 * |      15      |           330            |
 * |      16      |           495            |
 * |      17      |           495            |
 * |      18      |           715            |
 * |      19      |           715            |
 * |      20      |           1001           |
 */
class CubatureManager
{
private:
    DynRankView m_cubature_points;      ///< Multi-dimensional array holding cubature points (integration points).
    DynRankView m_cubature_weights;     ///< Multi-dimensional array holding cubature weights.
    unsigned short m_count_cubature_points; ///< Number of cubature points.
    unsigned short m_count_basis_functions; ///< Number of basis functions.

    /// @brief Initializes the cubature points and weights for the given cell type and polynomial order.
    void _initializeCubature(CellType cell_type, short desired_accuracy, short polynom_order);

public:
    /**
     * @brief Constructor for the CubatureManager class.
     *
     * Initializes the CubatureManager with the specified cell type, polynomial order, and desired accuracy.
     * These parameters are validated and then used to determine the number of cubature points and their
     * corresponding weights.
     *
     * @param cell_type The type of the cell (e.g., Tetrahedron, Hexahedron) for which cubature is needed.
     * @param desired_accuracy The desired accuracy for the cubature (default is defined by FEM limits).
     * @param polynom_order The polynomial order to be used in the basis (default is defined by FEM limits).
     *
     * @throw std::invalid_argument If the provided cell type is not supported.
     * @throw std::underflow_error If the polynomial order or desired accuracy is negative.
     * @throw std::invalid_argument If the polynomial order or desired accuracy is equal to zero.
     * @throw std::overflow_error If the polynomial order or desired accuracy exceeds the maximum allowable value.
     */
    CubatureManager(CellType cell_type, short desired_accuracy, short polynom_order);

    constexpr auto const &getCubaturePoints() const { return m_cubature_points; }
    constexpr auto const &getCubatureWeights() const { return m_cubature_weights; }
    constexpr unsigned short getCountCubaturePoints() const { return m_count_cubature_points; }
    constexpr unsigned short getCountBasisFunctions() const { return m_count_basis_functions; }
};

#endif // !CUBATUREMANAGER_HPP
