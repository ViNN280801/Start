#ifndef CELLSELECTOREXCEPTION_HPP
#define CELLSELECTOREXCEPTION_HPP

#include <exception>
#include <string>

#include "Utilities/Utilities.hpp"

#define CELLSELECTOR_UNSUPPORTED_CELL_TYPE_ERR "Unsupported cell type"
#define CELLSELECTOR_INVALID_ENUM_TYPE_ERR "Input is not an enum of type 'CellType'. Please check input"

#define THROW_CELL_SELECTOR_EXCEPTION() throw CellSelectorException()

/**
 * @class CellSelectorException
 * @brief Custom exception class for errors encountered in the CellSelector class.
 *
 * The CellSelectorException class is a custom exception type that extends the standard
 * `std::exception`. It allows for more informative error messages related to the
 * `CellSelector` class, such as handling unsupported cell types in finite element method (FEM) applications.
 */
class CellSelectorException : public std::exception
{
private:
    std::string m_message; ///< The error message to be displayed when the exception is thrown.

public:
    /**
     * @brief Constructs a new CellSelectorException with a specified error message.
     * @param message The error message that will be associated with the exception.
     *
     * This constructor initializes the exception with a custom error message, which can
     * later be retrieved by the `what()` method.
     */
    explicit CellSelectorException(std::string const &message = CELLSELECTOR_UNSUPPORTED_CELL_TYPE_ERR);

    /**
     * @brief Retrieves the error message associated with the exception.
     * @return A C-style string describing the error message.
     *
     * This method returns the error message that was provided when the exception was constructed.
     * It overrides the `what()` method from the base `std::exception` class, ensuring that
     * the error message can be easily accessed when the exception is caught.
     */
    const char *what() const noexcept override;
};

#endif // !CELLSELECTOREXCEPTION_HPP
