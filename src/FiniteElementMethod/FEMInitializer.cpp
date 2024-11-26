#include "FiniteElementMethod/FEMInitializer.hpp"
#include "FiniteElementMethod/BoundaryConditions/BoundaryConditionsManager.hpp"

FEMInitializer::FEMInitializer(ConfigParser const &config)
{
    // Assembling global stiffness matrix from the mesh file.
    m_assembler = std::make_shared<GSMAssembler>(config.getMeshFilename(), CellType::Tetrahedron, config.getDesiredCalculationAccuracy(), FEM_LIMITS_DEFAULT_POLYNOMIAL_ORDER);

    // Creating cubic grid for the tetrahedron mesh.
    m_cubicGrid = std::make_shared<CubicGrid>(m_assembler->getMeshManager(), config.getEdgeSize());

    // Setting boundary conditions.
    for (auto const &[nodeIds, value] : config.getBoundaryConditions())
        for (GlobalOrdinal nodeId : nodeIds)
            m_boundaryConditions[nodeId] = value;
    BoundaryConditionsManager::set(m_assembler->getGlobalStiffnessMatrix(), FEM_LIMITS_DEFAULT_POLYNOMIAL_ORDER, m_boundaryConditions);

    // Initializing the solution vector.
    m_solutionVector = std::make_shared<VectorManager>(m_assembler->getRows());
    m_solutionVector->clear();
}
