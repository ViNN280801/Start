#ifndef CELL_SELECTOR_HPP
#define CELL_SELECTOR_HPP

#include "FiniteElementMethod/FEMExceptions.hpp"
#include "FiniteElementMethod/Cell/CellType.hpp"
#include "FiniteElementMethod/FEMTypes.hpp"

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
    static shards::CellTopology get(CellType cellType);
};

#endif // !CELL_SELECTOR_HPP
