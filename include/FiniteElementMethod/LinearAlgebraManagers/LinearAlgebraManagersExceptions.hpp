#ifndef LINEAR_ALGEBRA_MANAGERS_EXCEPTIONS_HPP
#define LINEAR_ALGEBRA_MANAGERS_EXCEPTIONS_HPP

#include "Utilities/ExceptionMacros.hpp"

// ****************************** Base exceptions ************************************** //
START_DEFINE_EXCEPTION(LinearAlgebraManagersBaseException, std::runtime_error)
START_DEFINE_EXCEPTION(LinearAlgebraManagersUnknownException, LinearAlgebraManagersBaseException)
// ************************************************************************************* //

// ****************************** Matrix exceptions ************************************ //
START_DEFINE_EXCEPTION(MatrixManagerBaseException, LinearAlgebraManagersBaseException)
START_DEFINE_EXCEPTION(MatrixManagerEntriesEmptyException, std::invalid_argument)
START_DEFINE_EXCEPTION(MatrixManagerEntriesNegativeIndicesException, std::out_of_range)
START_DEFINE_EXCEPTION(MatrixManagerEntriesNotFoundException, std::out_of_range)
// ************************************************************************************* //

#endif // !LINEAR_ALGEBRA_MANAGERS_EXCEPTIONS_HPP
