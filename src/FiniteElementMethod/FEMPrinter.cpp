#include "FiniteElementMethod/FEMPrinter.hpp"

void FEMPrinter::printGraph(Teuchos::RCP<Tpetra::CrsGraph<>> const &graph)
{
    try
    {
        std::cout << "\n\nGraph data:\n";
        Teuchos::RCP<MapType const> rowMap{graph->getRowMap()};
        size_t numLocalRows{rowMap->getGlobalNumElements()};

        for (size_t i{}; i < numLocalRows; ++i)
        {
            GlobalOrdinal globalRow{rowMap->getGlobalElement(i)};
            size_t numEntries{graph->getNumEntriesInGlobalRow(globalRow)};
            TpetraMatrixType::nonconst_global_inds_host_view_type indices("ind", numEntries);

            size_t numIndices;
            graph->getGlobalRowCopy(globalRow, indices, numIndices);

            // Print row and its connections
            std::cout << "Row " << globalRow << ": ";
            for (size_t j{}; j < numIndices; ++j)
                std::cout << indices[j] << " ";
            std::cout << std::endl;
        }
    }
    catch (std::exception const &ex)
    {
        ERRMSG(ex.what());
    }
    catch (...)
    {
        ERRMSG("Unknown error");
    }
}

void FEMPrinter::printVector(Teuchos::RCP<TpetraVectorType> vector)
{
    Commutator comm{Tpetra::getDefaultComm()};

    int myRank{comm->getRank()};
    int numProcs{comm->getSize()};

    // Synchronize all processes before printing.
    comm->barrier();
    for (int proc{}; proc < numProcs; ++proc)
    {
        if (myRank == proc)
        {
            // Only the current process prints its portion of the vector.
            std::cout << std::format("Process {}\n", myRank);

            // Printing using describe() for detailed information.
            Teuchos::RCP<Teuchos::FancyOStream> out{Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout))};
            vector->describe(*out, Teuchos::VERB_EXTREME);

            // Printing individual elements
            auto vecView{vector->getLocalViewHost(Tpetra::Access::ReadOnly)};
            auto vecData{vecView.data()};
            size_t localLength{vector->getLocalLength()};
            for (size_t i{}; i < localLength; ++i)
                std::cout << std::format("Element {}: {}\n", i, vecData[i]);
        }
        // Synchronize before the next process starts printing.
        comm->barrier();
    }
    // Final barrier to ensure printing is finished before proceeding.
    comm->barrier();
}

void FEMPrinter::printMatrix(Teuchos::RCP<TpetraMatrixType> matrix)
{
    if (matrix->getGlobalNumEntries() == 0)
    {
        WARNINGMSG("Matrix is empty, nothing to print");
        return;
    }

    try
    {
        Commutator comm{Tpetra::getDefaultComm()};

        auto myRank{comm->getRank()};
        auto numProcs{comm->getSize()};

        // Loop over all processes in sequential order.
        comm->barrier();
        for (int proc{}; proc < numProcs; ++proc)
        {
            if (myRank == proc)
            {
                // Print the matrix entries for the current process.
                auto rowMap{matrix->getRowMap()};
                size_t localNumRows{rowMap->getLocalNumElements()};

                for (size_t i{}; i < localNumRows; ++i)
                {
                    GlobalOrdinal globalRow{rowMap->getGlobalElement(i)};
                    size_t numEntries{matrix->getNumEntriesInGlobalRow(globalRow)};

                    TpetraMatrixType::nonconst_global_inds_host_view_type indices("ind", numEntries);
                    TpetraMatrixType::nonconst_values_host_view_type values("val", numEntries);
                    size_t checkNumEntries{};

                    matrix->getGlobalRowCopy(globalRow, indices, values, checkNumEntries);

                    std::cout << std::format("Row {}: ", globalRow);
                    for (size_t k{}; k < checkNumEntries; ++k)
                        std::cout << "(" << indices[k] << ", " << values[k] << ") ";
                    std::endl(std::cout);
                }
            }
            // Synchronize all processes.
            comm->barrier();
        }
        comm->barrier();
    }
    catch (std::exception const &ex)
    {
        ERRMSG(ex.what());
    }
    catch (...)
    {
        ERRMSG("Unknown error was occured while printing global stiffness matrix");
    }
}
