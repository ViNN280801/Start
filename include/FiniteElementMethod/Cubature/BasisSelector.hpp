#ifndef BASISSELECTOR_HPP
#define BASISSELECTOR_HPP

#include <memory>

#include "CellSelectorException.hpp"
#include "CellType.hpp"
#include "FEMTypes.hpp"

/**
 * @class BasisSelector
 * @brief This class provides a static method to select the appropriate basis for different cell types.
 *
 * The method utilizes `CellType` to determine the basis function required for the finite element method
 * and supports various polynomial orders for the chosen basis. If an unsupported cell type is provided,
 * an exception is thrown.
 */
class BasisSelector
{
public:
    /**
     * @brief Selects the appropriate basis for the given cell type and polynomial order.
     *
     * @tparam DeviceType The type of device (e.g., CPU, GPU) for which the basis will be used.
     * It must satisfy a specific concept (e.g., be a valid execution space type).
     * @param cellType The type of the cell (e.g., Triangle, Tetrahedron, etc.).
     * @param polynom_order The polynomial order to be used in the basis.
     * This value defines the degree of the basis functions.
     * @return The selected basis corresponding to the cell type and polynomial order.
     * @throw CellSelectorException If the cell type is not supported.
     */
    template <DeviceTypeConcept DeviceType>
    static std::unique_ptr<Intrepid2::Basis<DeviceType>> get(CellType cellType, int polynom_order)
    {
        static_assert(std::is_enum_v<CellType>, "CELLSELECTOR_INVALID_ENUM_TYPE_ERR: cellType must be an enum");

        if (polynom_order < 0)
            throw std::runtime_error("Polynom order can't be negative");
        if (polynom_order == 0)
            throw std::runtime_error("Polynom order can't be equal to 0");

        switch (cellType)
        {
        case CellType::Triangle:
            return std::make_unique<Intrepid2::Basis_HGRAD_TRI_Cn_FEM<DeviceType>>(polynom_order);
        case CellType::Tetrahedron:
            return std::make_unique<Intrepid2::Basis_HGRAD_TET_Cn_FEM<DeviceType>>(polynom_order);
        case CellType::Pyramid:
            WARNINGMSG("Pyramid cells supports only 1st polynom order");
            return std::make_unique<Intrepid2::Basis_HGRAD_PYR_C1_FEM<DeviceType>>();
        case CellType::Wedge:
            WARNINGMSG("Wedge cells supports only 1st and 2nd polynom order, using 1st by default");
            return std::make_unique<Intrepid2::Basis_HGRAD_WEDGE_C1_FEM<DeviceType>>();
        case CellType::Hexahedron:
            WARNINGMSG("Hexahedron cells supports only 1st and 2nd polynom order, using 1st by default");
            return std::make_unique<Intrepid2::Basis_HGRAD_HEX_C1_FEM<DeviceType>>();
        default:
            THROW_CELL_SELECTOR_EXCEPTION();
        }
    }
};

#endif // !BASISSELECTOR_HPP
