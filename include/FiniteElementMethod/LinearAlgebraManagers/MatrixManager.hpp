#ifndef MATRIXMANAGER_HPP
#define MATRIXMANAGER_HPP

#include "FiniteElementMethod/FEMTypes.hpp"

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
    bool m_is_default_initialized{false};    ///< Flag that shows is the instance was initialized with empty default ctor or not.

public:
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

    /// @brief Ctor that does nothing. Needs for manually initialize map and matrix.
    explicit MatrixManager();

    /**
     * @brief Sets the map for the MatrixManager object.
     *
     * This method assigns a new map to the MatrixManager. It can only be used when
     * the MatrixManager was created using the default constructor. If any other
     * constructor was used (where the matrix or map was already initialized), this
     * method will throw an exception.
     *
     * WARNING: Available only when instance initialized with empty ctor.
     *
     * @param map The map to be assigned to the MatrixManager.
     * @throws std::logic_error If the MatrixManager was not created using the default constructor.
     */
    void setMap(Teuchos::RCP<MapType> map);

    /**
     * @brief Sets the matrix for the MatrixManager object.
     *
     * This method assigns a new matrix to the MatrixManager. Similar to setMap, this method
     * can only be used when the MatrixManager was created using the default constructor.
     * It will throw an exception if the manager was initialized through other constructors.
     *
     * WARNING: Available only when instance initialized with empty ctor.
     *
     * @param matrix The matrix to be assigned to the MatrixManager.
     * @throws std::logic_error If the MatrixManager was not created using the default constructor.
     */
    void setMatrix(Teuchos::RCP<TpetraMatrixType> matrix);

    /**
     * @brief Constructor for MatrixManager.
     *
     * Initializes the matrix with the given number of rows and columns, based on a distributed map.
     *
     * @param num_rows Number of rows in the matrix.
     * @param num_cols Number of columns in the matrix.
     */
    MatrixManager(GlobalOrdinal num_rows, GlobalOrdinal num_cols);

    /**
     * @brief Getter for the matrix.
     *
     * Provides access to the CRS matrix managed by the MatrixManager.
     *
     * @return A reference-counted pointer to the Tpetra matrix.
     */
    constexpr Teuchos::RCP<TpetraMatrixType> const &get() const { return m_matrix; }

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
