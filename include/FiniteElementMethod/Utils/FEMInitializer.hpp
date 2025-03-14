#ifndef FEMINITIALIZER_HPP
#define FEMINITIALIZER_HPP

#include "FiniteElementMethod/FEMTypes.hpp"
#include "FiniteElementMethod/Assemblers/GSMAssembler.hpp"
#include "FiniteElementMethod/LinearAlgebraManagers/VectorManager.hpp"
#include "FiniteElementMethod/Utils/FEMLimits.hpp"
#include "Geometry/Mesh/Cubic/CubicGrid.hpp"
#include "Utilities/ConfigParser.hpp"

/**
 * @class FEMInitializer
 * @brief Initializes the Finite Element Method (FEM) components including the global stiffness matrix,
 *        cubic grid, boundary conditions, and solution vector.
 *
 * This class serves as the core initializer for FEM computations, setting up necessary matrices, grids,
 * and vectors based on the provided configuration.
 */
class FEMInitializer
{
private:
    std::shared_ptr<GSMAssembler> m_assembler;            ///< Assembler for the global stiffness matrix.
    std::shared_ptr<CubicGrid> m_cubicGrid;               ///< Cubic grid for managing the tetrahedron mesh.
    std::map<GlobalOrdinal, double> m_boundaryConditions; ///< Boundary conditions for the FEM simulation.
    std::shared_ptr<VectorManager> m_solutionVector;      ///< Solution vector for the equation \( A \mathbf{x} = \mathbf{b} \).

public:
    /**
     * @brief Constructs an FEMInitializer and sets up FEM components.
     *
     * This constructor initializes the global stiffness matrix, cubic grid, boundary conditions,
     * and solution vector based on the configuration provided.
     *
     * @param config Reference to a ConfigParser object containing the simulation configuration.
     */
    FEMInitializer(ConfigParser const &config);

    /**
     * @brief Retrieves a modifiable reference to the global stiffness matrix assembler.
     * @return Modifiable reference to the GSM assembler.
     */
    auto getGlobalStiffnessMatrixAssembler() const { return m_assembler; }

    /**
     * @brief Retrieves a modifiable reference to the solution vector (right-hand side of the equation).
     * @return Modifiable reference to the solution vector.
     */
    auto getEquationRHS() const { return m_solutionVector; }

    /**
     * @brief Retrieves a modifiable reference to the cubic grid for the mesh.
     * @return Modifiable reference to the cubic grid.
     */
    auto getCubicGrid() const { return m_cubicGrid; }

    /**
     * @brief Retrieves a modifiable reference to the boundary conditions.
     * @return Constant reference to the boundary conditions map.
     */
    auto getBoundaryConditions() const { return m_boundaryConditions; }
};

#endif // !FEMINITIALIZER_HPP
