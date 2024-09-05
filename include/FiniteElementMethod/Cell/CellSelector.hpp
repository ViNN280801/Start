#include <concepts>
#include <exception>
#include <string>

#include "Utilities/Utilities.hpp"
#include "FEMTypes.hpp"
#include "CellSelectorException.hpp"
#include "CellType.hpp"

/**
 * @class CellSelector
 * @brief A static class responsible for selecting and managing different types of volumetric (3D) cells in finite element mesh generation.
 *
 * The CellSelector class encapsulates the different types of 3D volumetric cells that can be used in finite element methods (FEM).
 * It provides a simple static method to retrieve the cell type based on the given input.
 */
class CellSelector
{
public:
    /**
     * @brief Static method to retrieve the cell topology based on the provided cell type.
     * @param cellType The type of 3D cell to be selected.
     * @return The selected CellType.
     *
     * This method returns the topology for the given cell type using compile-time checks to ensure correctness.
     */
    static shards::CellTopology getCellType(CellType cellType)
    {
        static_assert(std::is_enum_v<CellType>, CELLSELECTOR_INVALID_ENUM_TYPE_ERR);

        switch (cellType)
        {
        case CellType::Triangle:
            return shards::getCellTopologyData<shards::Triangle<3>>();
        case CellType::Pentagon:
            return shards::getCellTopologyData<shards::Pentagon<5>>();
        case CellType::Hexagon:
            return shards::getCellTopologyData<shards::Hexagon<6>>();
        case CellType::Tetrahedron:
            return shards::getCellTopologyData<shards::Tetrahedron<4>>();
        case CellType::Pyramid:
            return shards::getCellTopologyData<shards::Pyramid<5>>();
        case CellType::Wedge:
            return shards::getCellTopologyData<shards::Wedge<6>>();
        case CellType::Hexahedron:
            return shards::getCellTopologyData<shards::Hexahedron<8>>();
        default:
            THROW_CELL_SELECTOR_EXCEPTION();

            static_assert(std::is_same_v<decltype(cellType), CellType>, CELLSELECTOR_INVALID_ENUM_TYPE_ERR);
        }
    }
};
