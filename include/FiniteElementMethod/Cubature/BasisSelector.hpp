#ifndef BASISSELECTOR_HPP
#define BASISSELECTOR_HPP

#include <Intrepid2_CellTools.hpp>
#include <Intrepid2_DefaultCubatureFactory.hpp>
#include <Intrepid2_FunctionSpaceTools.hpp>
#include <Intrepid2_HGRAD_TET_C1_FEM.hpp>
#include <Intrepid2_HGRAD_TET_C2_FEM.hpp>
#include <Intrepid2_HGRAD_TET_Cn_FEM.hpp>

#include <memory>

#include "CellSelectorException.hpp"
#include "CellType.hpp"
#include "FEMCheckers.hpp"
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
#if __cplusplus >= 202002L
    template <DeviceTypeConcept DeviceType>
#else
    template <typename DeviceType, typename = std::enable_if_t<DeviceTypeConcept_v<DeviceType>>>
#endif
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
    static std::unique_ptr<Intrepid2::Basis<DeviceType>> get(CellType cellType, int polynom_order)
    {
        FEMCheckers::checkPolynomOrder(polynom_order);

        switch (cellType)
        {
        case CellType::Triangle:
            return std::make_unique<Intrepid2::Basis_HGRAD_TRI_Cn_FEM<DeviceType>>(polynom_order);
        case CellType::Tetrahedron:
            return std::make_unique<Intrepid2::Basis_HGRAD_TET_Cn_FEM<DeviceType>>(polynom_order);
        case CellType::Pyramid:
            if (polynom_order != 1)
                throw std::runtime_error("Pyramid cells only support 1st polynomial order.");
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
