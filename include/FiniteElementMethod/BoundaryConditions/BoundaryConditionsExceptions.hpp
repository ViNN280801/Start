#ifndef BOUNDARY_CONDITIONS_EXCEPTIONS_HPP
#define BOUNDARY_CONDITIONS_EXCEPTIONS_HPP

#include "Utilities/ExceptionMacros.hpp"

// ****************************** Base exceptions *************************************************** //
START_DEFINE_EXCEPTION(BoundaryConditionsBaseException, std::runtime_error)
START_DEFINE_EXCEPTION(BoundaryConditionsUnknownException, BoundaryConditionsBaseException)
// *************************************************************************************************** //

// ****************************** Matrix boundary conditions exceptions ****************************** //
START_DEFINE_EXCEPTION(MatrixBoundaryConditionsBaseException, BoundaryConditionsBaseException)
START_DEFINE_EXCEPTION(MatrixBoundaryConditionsSettingException, std::logic_error)
START_DEFINE_EXCEPTION(MatrixBoundaryConditionsUnknownException, BoundaryConditionsUnknownException)
START_DEFINE_EXCEPTION(MatrixBoundaryConditionsInputParameterEmptyException, MatrixBoundaryConditionsBaseException)
START_DEFINE_EXCEPTION(MatrixBoundaryConditionsInputParameterInvalidException, std::invalid_argument)
START_DEFINE_EXCEPTION(MatrixBoundaryConditionsInputParameterOutOfRangeException, std::out_of_range)
// *************************************************************************************************** //

// ****************************** Vector boundary conditions exceptions ****************************** //
START_DEFINE_EXCEPTION(VectorBoundaryConditionsBaseException, BoundaryConditionsBaseException)
START_DEFINE_EXCEPTION(VectorBoundaryConditionsSettingException, std::logic_error)
START_DEFINE_EXCEPTION(VectorBoundaryConditionsUnknownException, BoundaryConditionsUnknownException)
START_DEFINE_EXCEPTION(VectorBoundaryConditionsNodeIDOutOfRangeException, std::out_of_range)
// *************************************************************************************************** //

#endif // !BOUNDARY_CONDITIONS_EXCEPTIONS_HPP
