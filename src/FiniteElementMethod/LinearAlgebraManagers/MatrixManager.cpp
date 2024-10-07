#include <algorithm>
#include <ranges>

#include "FiniteElementMethod/FEMCheckers.hpp"
#include "FiniteElementMethod/LinearAlgebraManagers/MatrixManager.hpp"

MatrixManager::MatrixManager(std::span<MatrixEntry const> matrix_entries) : m_entries(matrix_entries.begin(), matrix_entries.end())
{
    if (matrix_entries.empty())
        throw std::invalid_argument("Matrix entries can't be empty");

    if (std::ranges::any_of(matrix_entries, [](MatrixEntry const &entry)
                            { return entry.row < 0 || entry.col < 0; }))
        throw std::out_of_range("Matrix entries cannot have negative row or column indices");

    // 1. Collect global row indices
    std::set<GlobalOrdinal> globalRowIndices;
    for (auto const &entry : m_entries)
        globalRowIndices.insert(entry.row);

    // 2. Create the Tpetra Map with local elements
    Teuchos::Array<GlobalOrdinal> myGlobalElements(globalRowIndices.begin(), globalRowIndices.end());
    short indexBase{};
    m_map = Teuchos::rcp(new MapType(Teuchos::OrdinalTraits<GlobalOrdinal>::invalid(),
                                     myGlobalElements(), indexBase, Tpetra::getDefaultComm()));

    // 3. Build the graph entries
    std::map<GlobalOrdinal, std::set<GlobalOrdinal>> graphEntries;
    for (const auto &entry : m_entries)
    {
        graphEntries[entry.row].insert(entry.col);
    }

    // 4. Initialize the CrsGraph
    size_t numLocalRows = m_map->getLocalNumElements();
    std::vector<size_t> numEntriesPerRow(numLocalRows, 0);

    auto globalToLocalRow = [this](GlobalOrdinal globalRow)
    {
        return m_map->getLocalElement(globalRow);
    };

    for (const auto &rowEntry : graphEntries)
    {
        auto localIndex {globalToLocalRow(rowEntry.first)};
        if (localIndex == Teuchos::OrdinalTraits<LocalOrdinal>::invalid())
        {
            continue; // Skip rows not owned by this process
        }
        numEntriesPerRow[localIndex] = rowEntry.second.size();
    }

    Teuchos::RCP<Tpetra::CrsGraph<>> graph{Teuchos::rcp(new Tpetra::CrsGraph<>(m_map, Teuchos::ArrayView<size_t const>(numEntriesPerRow.data(), numEntriesPerRow.size())))};
    for (auto const &rowEntries : graphEntries)
    {
        if (!m_map->isNodeGlobalElement(rowEntries.first))
            continue; // Skip rows not owned by this process
        std::vector<GlobalOrdinal> columns(rowEntries.second.begin(), rowEntries.second.end());
        Teuchos::ArrayView<const GlobalOrdinal> colsView(columns.data(), columns.size());
        graph->insertGlobalIndices(rowEntries.first, colsView);
    }
    graph->fillComplete();

    // 5. Initialize the matrix
    m_matrix = Teuchos::rcp(new TpetraMatrixType(graph));

    // 6. Insert matrix entries
    for (const auto &entry : m_entries)
    {
        if (!m_map->isNodeGlobalElement(entry.row))
        {
            continue; // Skip entries not owned by this process
        }
        Teuchos::ArrayView<const GlobalOrdinal> colsView(&entry.col, 1);
        Teuchos::ArrayView<const Scalar> valsView(&entry.value, 1);
        m_matrix->sumIntoGlobalValues(entry.row, colsView, valsView);
    }

    // 7. Finalize the matrix
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
