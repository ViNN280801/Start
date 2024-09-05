#include "FiniteElementMethod/BoundaryConditions/BoundaryConditionsManager.hpp"
#include "FiniteElementMethod/BoundaryConditions/MatrixBoundaryConditionsManager.hpp"
#include "FiniteElementMethod/BoundaryConditions/VectorBoundaryConditionManager.hpp"

void BoundaryConditionsManager::set(Teuchos::RCP<TpetraVectorType> vector, short polynom_order, std::map<GlobalOrdinal, Scalar> const &boundary_conditions)
{
    VectorBoundaryConditionsManager::set(vector, polynom_order, boundary_conditions);
}

void BoundaryConditionsManager::set(Teuchos::RCP<TpetraMatrixType> matrix, short polynom_order, std::map<GlobalOrdinal, Scalar> const &boundary_conditions)
{
    MatrixBoundaryConditionsManager::set(matrix, polynom_order, boundary_conditions);
}
