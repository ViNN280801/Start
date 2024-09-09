#ifndef FEMPRINTER_HPP
#define FEMPRINTER_HPP

#include "FEMTypes.hpp"
#include "GSMatrixAssemblier.hpp"
#include "Utilities/Utilities.hpp"

/**
 * @class FEMPrinter
 * @brief A utility class for developers to print the contents of Tpetra graph, vector, and CRS matrix.
 *
 * The FEMPrinter class provides static methods to print detailed information about Tpetra structures,
 * such as graphs, vectors, and matrices. These methods allow developers to inspect the internal data
 * structures of Tpetra to aid in debugging and development of finite element applications.
 */
class FEMPrinter
{
public:
    /**
     * @brief Prints the details of a Tpetra graph.
     *
     * This method prints the global rows of a given Tpetra `CrsGraph` and lists its connections (entries)
     * for each row. It displays the data in a readable format for debugging and analysis.
     *
     * @param graph A reference-counted pointer (RCP) to a Tpetra `CrsGraph`.
     * @throws std::exception If an error occurs during the operation, an error message is displayed.
     * @throws ... If an unknown error occurs, an error message is displayed.
     */
    static void printGraph(Teuchos::RCP<Tpetra::CrsGraph<>> const &graph);

    /**
     * @brief Prints the contents of a Tpetra vector.
     *
     * This method prints the elements of a Tpetra `Vector` from each process in the communicator.
     * The output includes both a detailed description using `describe()` and individual elements
     * from the local portion of the vector for each process.
     *
     * @param vector A reference-counted pointer (RCP) to a Tpetra `Vector`.
     * @throws std::exception If an error occurs during the operation, an error message is displayed.
     * @throws ... If an unknown error occurs, an error message is displayed.
     */
    static void printVector(Teuchos::RCP<TpetraVectorType> vector);

    /**
     * @brief Prints the contents of a Tpetra CRS matrix.
     *
     * This method prints the global row entries and values of a Tpetra `CrsMatrix`. It loops through
     * all processes, ensuring each process prints its local portion of the matrix, including global
     * row indices and the associated non-zero matrix entries.
     *
     * @param matrix A reference-counted pointer (RCP) to a Tpetra `CrsMatrix`.
     * @throws std::exception If an error occurs during the operation, an error message is displayed.
     * @throws ... If an unknown error occurs, an error message is displayed.
     */
    static void printMatrix(Teuchos::RCP<TpetraMatrixType> matrix);
};

#endif // !FEMPRINTER_HPP
