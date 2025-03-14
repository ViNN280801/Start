#ifndef FEM_UTILS_EXCEPTIONS_HPP
#define FEM_UTILS_EXCEPTIONS_HPP

#include "Utilities/ExceptionMacros.hpp"

// ****************************** Base exceptions ************************************** //
START_DEFINE_EXCEPTION(FEMUtilsBaseException, std::runtime_error)
START_DEFINE_EXCEPTION(FEMUtilsUnknownException, FEMUtilsBaseException)
// ************************************************************************************* //

// ****************************** FEMCheckers exceptions ****************************** //
START_DEFINE_EXCEPTION(FEMCheckersBaseException, FEMUtilsBaseException)
START_DEFINE_EXCEPTION(FEMCheckersUnsupportedCellTypeException, FEMCheckersBaseException)
START_DEFINE_EXCEPTION(FEMCheckersUnderflowDesiredAccuracyException, FEMCheckersBaseException)
START_DEFINE_EXCEPTION(FEMCheckersOverflowDesiredAccuracyException, FEMCheckersBaseException)
START_DEFINE_EXCEPTION(FEMCheckersUnsupportedDesiredAccuracyException, FEMCheckersBaseException)
START_DEFINE_EXCEPTION(FEMCheckersUnsupportedPolynomOrderException, FEMCheckersBaseException)
START_DEFINE_EXCEPTION(FEMCheckersIndexOutOfRangeException, std::logic_error)
START_DEFINE_EXCEPTION(FEMCheckersMatrixMismatchException, FEMCheckersBaseException)
START_DEFINE_EXCEPTION(FEMCheckersBoundaryConditionOutOfRangeException, std::out_of_range)
START_DEFINE_EXCEPTION(FEMCheckersUnknownException, FEMCheckersBaseException)
// *************************************************************************************//

// ****************************** FEMPrinter exceptions ****************************** //
START_DEFINE_EXCEPTION(FEMPrinterBaseException, FEMUtilsBaseException)
START_DEFINE_EXCEPTION(FEMUtilsPrintGraphException, FEMPrinterBaseException)
START_DEFINE_EXCEPTION(FEMUtilsPrintMatrixException, FEMPrinterBaseException)
// *********************************************************************************** //

#endif // !FEM_UTILS_EXCEPTIONS_HPP
