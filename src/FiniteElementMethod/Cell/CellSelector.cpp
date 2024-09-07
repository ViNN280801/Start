#include "FiniteElementMethod/Cell/CellSelector.hpp"

shards::CellTopology CellSelector::get(CellType cellType)
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
