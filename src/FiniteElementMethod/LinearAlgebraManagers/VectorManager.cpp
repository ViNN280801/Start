#include "FiniteElementMethod/LinearAlgebraManagers/VectorManager.hpp"
#include "FiniteElementMethod/FEMCheckers.hpp"

VectorManager::VectorManager(size_t size)
    : m_map(new MapType(size, 0, Tpetra::getDefaultComm())), // 0 here is the index base.
      m_vector(Teuchos::rcp(new TpetraVectorType(m_map)))
{
}

size_t VectorManager::size() const { return m_vector->getGlobalLength(); }

void VectorManager::clear() { m_vector->putScalar(0.0); }

void VectorManager::randomize() { m_vector->randomize(); }

Scalar &VectorManager::at(GlobalOrdinal i)
{
    FEMCheckers::checkIndex(i, size());
    return m_vector->get1dViewNonConst()[i];
}

Scalar const &VectorManager::at(GlobalOrdinal i) const
{
    FEMCheckers::checkIndex(i, size());
    return m_vector->get1dView()[i];
}

Scalar &VectorManager::operator[](GlobalOrdinal i) { return at(i); }

Scalar const &VectorManager::operator[](GlobalOrdinal i) const { return at(i); }
