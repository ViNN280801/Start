#include "FiniteElementMethod/LinearAlgebraManagers/MatrixManager.hpp"
#include "FiniteElementMethod/FEMCheckers.hpp"

MatrixManager::MatrixManager(std::vector<MatrixEntry> const &matrix_entries) : m_entries(matrix_entries)
{
    // 1. Getting unique global entries.
    std::map<GlobalOrdinal, std::set<GlobalOrdinal>> graphEntries;
    for (auto const &entry : m_entries)
        graphEntries[entry.row].insert(entry.col);

    // 2. Initializing all necessary variables.
    short indexBase{};
    auto countGlobalNodes{graphEntries.size()};

    // 3. Initializing tpetra map.
    m_map = Teuchos::rcp(new MapType(countGlobalNodes, indexBase, Tpetra::getDefaultComm()));

    // 4. Initializing tpetra graph.
    std::vector<size_t> numEntriesPerRow(countGlobalNodes);
    for (auto const &rowEntry : graphEntries)
        numEntriesPerRow.at(m_map->getLocalElement(rowEntry.first)) = rowEntry.second.size();

    Teuchos::RCP<Tpetra::CrsGraph<>> graph{Teuchos::rcp(new Tpetra::CrsGraph<>(m_map, Teuchos::ArrayView<size_t const>(numEntriesPerRow.data(), numEntriesPerRow.size())))};
    for (auto const &rowEntries : graphEntries)
    {
        std::vector<GlobalOrdinal> columns(rowEntries.second.begin(), rowEntries.second.end());
        Teuchos::ArrayView<GlobalOrdinal const> colsView(columns.data(), columns.size());
        graph->insertGlobalIndices(rowEntries.first, colsView);
    }
    graph->fillComplete();

    // 5. Initializing matrix.
    m_matrix = Teuchos::rcp(new TpetraMatrixType(graph));

    // 6. Finalizing work with matrix.
    m_matrix->fillComplete();
}

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
