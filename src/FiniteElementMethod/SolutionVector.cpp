#include "FiniteElementMethod/SolutionVector.hpp"
#include "FiniteElementMethod/BoundaryConditions/BoundaryConditionsManager.hpp"
#include "FiniteElementMethod/FEMCheckers.hpp"
#include "FiniteElementMethod/FEMPrinter.hpp"
#include "Utilities/Utilities.hpp"

SolutionVector::SolutionVector(size_t size, short polynom_order)
    : m_map(new MapType(size, 0, Tpetra::getDefaultComm())), // 0 here is the index base.
      m_solution_vector(Teuchos::rcp(new TpetraVectorType(m_map))),
      m_polynom_order(polynom_order)
{
    FEMCheckers::checkPolynomOrder(polynom_order);
}

void SolutionVector::setBoundaryConditions(std::map<GlobalOrdinal, Scalar> const &boundaryConditions) { BoundaryConditionsManager::set(m_solution_vector, m_polynom_order, boundaryConditions); }

size_t SolutionVector::size() const { return m_solution_vector->getGlobalLength(); }

void SolutionVector::randomize() { m_solution_vector->randomize(); }

void SolutionVector::clear() { m_solution_vector->putScalar(0.0); }

void SolutionVector::print() const { FEMPrinter::printVector(m_solution_vector); }
