#include "FiniteElementMethod/LinearAlgebraManagers/MatrixManager.hpp"
#include "FiniteElementMethod/FEMCheckers.hpp"

MatrixManager::MatrixManager() : m_is_default_initialized(true) {}

void MatrixManager::setMap(Teuchos::RCP<MapType> map)
{
    if (!m_is_default_initialized)
        throw std::logic_error("setMap can only be used with the default constructor");
    m_map = map;
}

void MatrixManager::setMatrix(Teuchos::RCP<TpetraMatrixType> matrix)
{
    if (!m_is_default_initialized)
        throw std::logic_error("setMatrix can only be used with the default constructor");
    m_matrix = matrix;
}

MatrixManager::MatrixManager(GlobalOrdinal num_rows, GlobalOrdinal num_cols)
    : m_map(Teuchos::rcp(new MapType(num_rows, 0, Tpetra::getDefaultComm()))),
      m_matrix(Teuchos::rcp(new TpetraMatrixType(m_map, num_cols))) {}

size_t MatrixManager::rows() const { return m_matrix->getGlobalNumRows(); }

size_t MatrixManager::cols() const { return m_matrix->getGlobalNumCols(); }

bool MatrixManager::empty() const { return m_matrix->getGlobalNumEntries() == 0; }

Scalar &MatrixManager::at(GlobalOrdinal row, GlobalOrdinal col)
{
    FEMCheckers::checkIndex(row, rows(), "Row ");
    FEMCheckers::checkIndex(col, cols(), "Column ");

    // Get the number of entries in specified row and check it.
    size_t numEntries{m_matrix->getNumEntriesInGlobalRow(row)};
    if (numEntries == 0)
        throw std::out_of_range("Row has no entries");

    // Create views for indices and values
    TpetraMatrixType::nonconst_global_inds_host_view_type indices("indices", numEntries);
    TpetraMatrixType::nonconst_values_host_view_type values("values", numEntries);

    size_t actualNumEntries{};

    // Get the global row copy with indices and values
    m_matrix->getGlobalRowCopy(row, indices, values, actualNumEntries);

    // Search for the column
    for (size_t i{}; i < actualNumEntries; ++i)
        if (indices[i] == col)
            return values[i];

    throw std::out_of_range("Column not found in row");
}

Scalar const &MatrixManager::at(GlobalOrdinal row, GlobalOrdinal col) const
{
    FEMCheckers::checkIndex(row, rows(), "Row ");
    FEMCheckers::checkIndex(col, cols(), "Column ");

    size_t numEntries{m_matrix->getNumEntriesInGlobalRow(row)};
    if (numEntries == 0)
        throw std::out_of_range("Row has no entries");

    TpetraMatrixType::nonconst_global_inds_host_view_type indices("indices", numEntries);
    TpetraMatrixType::nonconst_values_host_view_type values("values", numEntries);

    size_t actualNumEntries{};
    m_matrix->getGlobalRowCopy(row, indices, values, actualNumEntries);

    for (size_t i{}; i < actualNumEntries; ++i)
        if (indices[i] == col)
            return values[i];

    throw std::out_of_range("Column not found in row");
}

Scalar &MatrixManager::operator()(GlobalOrdinal row, GlobalOrdinal col) { return at(row, col); }

Scalar const &MatrixManager::operator()(GlobalOrdinal row, GlobalOrdinal col) const { return at(row, col); }
