#ifndef MATRIXMANAGER_HPP
#define MATRIXMANAGER_HPP

#include "FiniteElementMethod/FEMTypes.hpp"

/**
 * @struct MatrixEntry
 * @brief Struct representing an entry in the global stiffness matrix.
 *
 * The MatrixEntry struct stores the row index, column index, and value of a single entry in the global matrix.
 */
struct MatrixEntry
{
    GlobalOrdinal row; ///< Global row index for the matrix entry.
    GlobalOrdinal col; ///< Global column index for the matrix entry.
    Scalar value;      ///< Value to be inserted at (row, col) in the global matrix.
};

/**
 * @class MatrixManager
 * @brief Class responsible for managing the global stiffness matrix in the equation Ax = b in FEM.
 *
 * The MatrixManager class provides methods to manage and manipulate matrices used in FEM computations.
 * It supports distributed computation by using Teuchos RCP (Reference-counted pointers) to manage memory
 * and facilitate parallel processing. The matrix is stored in compressed row storage (CRS) format.
 *
 * @note This class is designed to manage sparse matrices in a distributed parallel environment.
 */
class MatrixManager
{
private:
    Teuchos::RCP<MapType> m_map;             ///< A smart pointer managing the lifetime of a Map object, which defines the layout of distributed data across the processes in a parallel computation.
    Teuchos::RCP<TpetraMatrixType> m_matrix; ///< Smart pointer on the CRS matrix.
    std::vector<MatrixEntry> m_entries;      ///< Entries of the matrix: [row][col]=<value>. So-called "triplets": row number, column number and value.

public:
    /**
     * @brief Constructor for the MatrixManager class using a list of matrix entries.
     *
     * This constructor initializes the MatrixManager with a set of global stiffness matrix entries.
     * It processes the matrix entries (row, col, value) to build the Tpetra compressed row storage (CRS) matrix.
     * The constructor follows these steps:
     * - Collects unique global row-column entries to construct the CRS graph.
     * - Initializes the Tpetra map based on the global node count.
     * - Creates the Tpetra CRS graph using the number of entries per row.
     * - Inserts the global indices into the graph and finalizes it.
     * - Initializes the global stiffness matrix and prepares it for further operations.
     *
     * @param matrix_entries A vector of MatrixEntry structures containing the row, column, and value information
     *                       for the non-zero entries in the global stiffness matrix.
     *
     * @note This constructor assumes that the matrix is sparse and the number of non-zero entries is significantly
     *       smaller than the total number of possible entries in a full matrix.
     * @throws std::runtime_error if there is any issue during the matrix or graph construction process.
     */
    MatrixManager(std::vector<MatrixEntry> const &matrix_entries);

    /**
     * @brief Getter for the matrix.
     *
     * Provides access to the CRS matrix managed by the MatrixManager.
     *
     * @return A reference-counted pointer to the Tpetra matrix.
     */
    Teuchos::RCP<TpetraMatrixType> get() const { return m_matrix; }

    /**
     * @brief Retrieves the matrix entries stored in the MatrixManager.
     *
     * The matrix entries represent the global indices and corresponding values of the non-zero elements
     * in the global stiffness matrix. These entries are provided in a triplet format (row, col, value),
     * which is useful for constructing the sparse matrix or for retrieving and manipulating its values.
     *
     * @return A constant reference to the vector of matrix entries (row, col, value).
     */
    constexpr auto const &getMatrixEntries() const { return m_entries; }

    /**
     * @brief Returns the number of rows in the global matrix.
     * @return Number of rows in the matrix.
     */
    size_t rows() const;

    /**
     * @brief Returns the number of columns in the global matrix.
     * @return Number of columns in the matrix.
     */
    size_t cols() const;

    /**
     * @brief Checks whether the global stiffness matrix is empty.
     *
     * Determines if there are no entries in the matrix.
     *
     * @return True if the matrix is empty, false otherwise.
     */
    bool empty() const;

    /**
     * @brief Non-const access to matrix element at (row, col).
     *
     * This method provides non-const access to the matrix element at the specified row and column,
     * allowing modifications to the matrix value.
     *
     * @param row The global row index.
     * @param col The global column index.
     * @return A reference to the element at (row, col).
     * @throws std::out_of_range if the indices are out of bounds.
     */
    Scalar &at(GlobalOrdinal row, GlobalOrdinal col);

    /**
     * @brief Const access to matrix element at (row, col).
     *
     * This method provides read-only access to the matrix element at the specified row and column.
     *
     * @param row The global row index.
     * @param col The global column index.
     * @return A const reference to the element at (row, col).
     * @throws std::out_of_range if the indices are out of bounds.
     */
    Scalar const &at(GlobalOrdinal row, GlobalOrdinal col) const;

    /**
     * @brief Non-const access to matrix element using the call operator (row, col).
     *
     * This is a convenience method for accessing elements in the matrix with the () operator.
     * It allows modifying the element at the given row and column.
     *
     * @param row The local row index.
     * @param col The local column index.
     * @return A reference to the element at (row, col).
     */
    Scalar &operator()(GlobalOrdinal row, GlobalOrdinal col);

    /**
     * @brief Const access to matrix element using the call operator (row, col).
     *
     * This is a convenience method for accessing elements in the matrix with the () operator in a read-only manner.
     *
     * @param row The local row index.
     * @param col The local column index.
     * @return A const reference to the element at (row, col).
     */
    Scalar const &operator()(GlobalOrdinal row, GlobalOrdinal col) const;
};

#endif // !MATRIXMANAGER_HPP
