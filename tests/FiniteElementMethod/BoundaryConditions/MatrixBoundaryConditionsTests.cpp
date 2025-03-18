#include <fstream>
#include <functional>
#include <gtest/gtest.h>
#include <iostream>
#include <stdexcept>

#include "FiniteElementMethod/BoundaryConditions/MatrixBoundaryConditionsManager.hpp"
#include "FiniteElementMethod/FEMExceptions.hpp"

void supress_output(std::ostream &stream) { stream.setstate(std::ios_base::failbit); }
void restore_output(std::ostream &stream) { stream.clear(); }

void execute_without_output(std::function<void()> const &func, std::ostream &stream)
{
    supress_output(stream);
    func();
    restore_output(stream);
}

Teuchos::RCP<TpetraMatrixType> createTestMatrix(size_t rows, size_t cols)
{
    // 1. Creating map
    Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal>> map =
        Teuchos::rcp(new Tpetra::Map<LocalOrdinal, GlobalOrdinal>(rows, 0, Teuchos::DefaultComm<int>::getComm()));

    // 2. Creating graph for the matrix
    std::vector<size_t> numEntriesPerRow(rows, cols);
    Teuchos::RCP<Tpetra::CrsGraph<>> graph = Teuchos::rcp(new Tpetra::CrsGraph<>(map, Teuchos::ArrayView<size_t const>(numEntriesPerRow.data(), numEntriesPerRow.size())));

    // 3. Inserting indices in graph
    for (GlobalOrdinal i = 0; i < static_cast<GlobalOrdinal>(rows); ++i)
    {
        std::vector<GlobalOrdinal> indices(cols);
        for (GlobalOrdinal j = 0; j < static_cast<GlobalOrdinal>(cols); ++j)
        {
            indices[j] = j;
        }
        graph->insertGlobalIndices(i, Teuchos::ArrayView<GlobalOrdinal>(indices.data(), cols));
    }

    // 4. Finalizing graph filling
    graph->fillComplete();

    // 5. Creating matrix based on graph
    Teuchos::RCP<TpetraMatrixType> matrix = Teuchos::rcp(new TpetraMatrixType(graph));

    // 6. Filling matrix with values
    for (GlobalOrdinal i = 0; i < static_cast<GlobalOrdinal>(rows); ++i)
    {
        Teuchos::Array<GlobalOrdinal> indices(cols);
        Teuchos::Array<Scalar> values(cols, static_cast<Scalar>(1.0));

        for (GlobalOrdinal j = 0; j < static_cast<GlobalOrdinal>(cols); ++j)
        {
            indices[j] = j;
        }

        matrix->replaceGlobalValues(i, indices(), values());
    }

    // 7. Ending of the matrix filling
    matrix->fillComplete();

    return matrix;
}

bool checkMatrixBoundaryConditions(Teuchos::RCP<TpetraMatrixType> matrix, std::map<GlobalOrdinal, Scalar> const &boundary_conditions, size_t polynom_order)
{
    for (auto const &[nodeInGmsh, value] : boundary_conditions)
    {
        for (size_t j{}; j < polynom_order; ++j)
        {
            GlobalOrdinal nodeID = (nodeInGmsh - 1) * polynom_order + j;
            size_t numEntries = matrix->getNumEntriesInGlobalRow(nodeID);
            TpetraMatrixType::nonconst_global_inds_host_view_type indices("indices", numEntries);
            TpetraMatrixType::nonconst_values_host_view_type values("values", numEntries);
            matrix->getGlobalRowCopy(nodeID, indices, values, numEntries);

            // Check that the diagonal is set to 'value' and the other elements are set to 0
            for (size_t i = 0; i < numEntries; ++i)
            {
                if (indices(i) == nodeID && values(i) != 1)
                {
                    return false;
                }
                else if (indices(i) != nodeID && values(i) != 0)
                {
                    return false;
                }
            }
        }
    }
    return true;
}

// Clean test: checking the setting of boundary conditions
TEST(MatrixBoundaryConditionsManagerTest, ValidBoundaryCondition)
{
    Teuchos::RCP<TpetraMatrixType> matrix = createTestMatrix(5, 5);
    std::map<GlobalOrdinal, Scalar> boundary_conditions{{1, 1.0}, {2, 0.0}};

    MatrixBoundaryConditionsManager manager;
    manager.set(matrix, 1, boundary_conditions);

    ASSERT_TRUE(checkMatrixBoundaryConditions(matrix, boundary_conditions, 1));
}

// Dirty Test: empty matrix
TEST(MatrixBoundaryConditionsManagerTest, EmptyMatrix)
{
    Teuchos::RCP<TpetraMatrixType> matrix;
    std::map<GlobalOrdinal, Scalar> boundary_conditions{{1, 1.0}, {2, 0.0}};

    execute_without_output([&]()
                           {
    MatrixBoundaryConditionsManager manager;
    EXPECT_NO_THROW(manager.set(matrix, 1, boundary_conditions)); }, std::cerr);
}

// Dirty Test: empty boundary conditions
TEST(MatrixBoundaryConditionsManagerTest, EmptyBoundaryConditions)
{
    Teuchos::RCP<TpetraMatrixType> matrix = createTestMatrix(500, 500);
    std::map<GlobalOrdinal, Scalar> boundary_conditions;

    execute_without_output([&]()
                           {
    MatrixBoundaryConditionsManager manager;
    EXPECT_NO_THROW(manager.set(matrix, 1, boundary_conditions)); }, std::cerr);
}

// Dirty test: node is out of range
TEST(MatrixBoundaryConditionsManagerTest, OutOfBoundsNodeID)
{
    Teuchos::RCP<TpetraMatrixType> matrix = createTestMatrix(5, 5);
    std::map<GlobalOrdinal, Scalar> boundary_conditions{{10, 1.0}, {11, 0.0}};

    execute_without_output([&]()
                           {
    MatrixBoundaryConditionsManager manager;
    EXPECT_THROW(manager.set(matrix, 1, boundary_conditions),
        MatrixBoundaryConditionsSettingException); }, std::cerr);
}

// Dirty test: boundary conditions with a polynomial of order 2
TEST(MatrixBoundaryConditionsManagerTest, PolynomialOrderTest)
{
    Teuchos::RCP<TpetraMatrixType> matrix = createTestMatrix(1000, 1000);
    std::map<GlobalOrdinal, Scalar> boundary_conditions{{1, 1.0}, {2, 0.0}};

    MatrixBoundaryConditionsManager manager;
    manager.set(matrix, 2, boundary_conditions);

    ASSERT_TRUE(checkMatrixBoundaryConditions(matrix, boundary_conditions, 2));
}
