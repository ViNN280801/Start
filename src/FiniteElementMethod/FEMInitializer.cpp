#include "FiniteElementMethod/FEMInitializer.hpp"
#include "FiniteElementMethod/BoundaryConditions/BoundaryConditionsManager.hpp"

FEMInitializer::FEMInitializer(ConfigParser const &config)
{
    // Assembling global stiffness matrix from the mesh file.
    m_assemblier = std::make_shared<GSMAssemblier>(config.getMeshFilename(), CellType::Tetrahedron, config.getDesiredCalculationAccuracy(), FEM_LIMITS_DEFAULT_POLYNOMIAL_ORDER);

    // Creating cubic grid for the tetrahedron mesh.
    m_cubicGrid = std::make_shared<CubicGrid>(m_assemblier->getMeshManager(), config.getEdgeSize());

    // Setting boundary conditions.
    for (auto const &[nodeIds, value] : config.getBoundaryConditions())
        for (GlobalOrdinal nodeId : nodeIds)
            m_boundaryConditions[nodeId] = value;
    BoundaryConditionsManager::set(m_assemblier->getGlobalStiffnessMatrix(), FEM_LIMITS_DEFAULT_POLYNOMIAL_ORDER, m_boundaryConditions);

    // Initializing the solution vector.
    m_solutionVector = std::make_shared<VectorManager>(m_assemblier->getRows());
    m_solutionVector->clear();
}
