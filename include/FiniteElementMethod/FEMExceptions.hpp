#ifndef FEM_EXCEPTIONS_HPP
#define FEM_EXCEPTIONS_HPP

#include "Utilities/ExceptionMacros.hpp"

// ****************************** Base exceptions ************************************** //
START_DEFINE_EXCEPTION(FEMBaseException, std::runtime_error)
START_DEFINE_EXCEPTION(FEMLogicErrorException, std::logic_error)
START_DEFINE_EXCEPTION(FEMInvalidArgumentErrorException, std::invalid_argument)
START_DEFINE_EXCEPTION(FEMOutOfRangeErrorException, std::out_of_range)
START_DEFINE_EXCEPTION(FEMUnderflowErrorException, std::underflow_error)
START_DEFINE_EXCEPTION(FEMOverflowErrorException, std::overflow_error)
START_DEFINE_EXCEPTION(FEMUnknownException, FEMBaseException)
// ************************************************************************************* //

// ****************************** GSM assembler exceptions ****************************** //
START_DEFINE_EXCEPTION(GSMAssemblerBaseException, FEMBaseException)
START_DEFINE_EXCEPTION(GSMAssemblerGettingMatrixEntriesException, FEMInvalidArgumentErrorException)
START_DEFINE_EXCEPTION(GSMAssemblerComputingLocalStiffnessMatricesException, FEMInvalidArgumentErrorException)
START_DEFINE_EXCEPTION(GSMAssemblerUnknownException, FEMUnknownException)
// ************************************************************************************* //

// ****************************** Matrix boundary conditions exceptions ****************************** //
START_DEFINE_EXCEPTION(MatrixBoundaryConditionsBaseException, FEMBaseException)
START_DEFINE_EXCEPTION(MatrixBoundaryConditionsSettingException, FEMLogicErrorException)
START_DEFINE_EXCEPTION(MatrixBoundaryConditionsUnknownException, FEMUnknownException)
START_DEFINE_EXCEPTION(MatrixBoundaryConditionsInputParameterEmptyException, MatrixBoundaryConditionsBaseException)
START_DEFINE_EXCEPTION(MatrixBoundaryConditionsInputParameterInvalidException, FEMInvalidArgumentErrorException)
START_DEFINE_EXCEPTION(MatrixBoundaryConditionsInputParameterOutOfRangeException, FEMOutOfRangeErrorException)
// *************************************************************************************************** //

// ****************************** Vector boundary conditions exceptions ****************************** //
START_DEFINE_EXCEPTION(VectorBoundaryConditionsBaseException, FEMBaseException)
START_DEFINE_EXCEPTION(VectorBoundaryConditionsSettingException, FEMLogicErrorException)
START_DEFINE_EXCEPTION(VectorBoundaryConditionsUnknownException, FEMUnknownException)
START_DEFINE_EXCEPTION(VectorBoundaryConditionsNodeIDOutOfRangeException, FEMOutOfRangeErrorException)
// *************************************************************************************************** //

// ****************************** Cell selector exceptions ****************************** //
START_DEFINE_EXCEPTION(CellSelectorBaseException, FEMBaseException)
START_DEFINE_EXCEPTION(CellSelectorUnsupportedCellTypeException, CellSelectorBaseException)
START_DEFINE_EXCEPTION(CellSelectorInvalidEnumTypeException, CellSelectorBaseException)
// ************************************************************************************* //

// ****************************** Basis selector exceptions ****************************** //
START_DEFINE_EXCEPTION(BasisSelectorBaseException, FEMBaseException)
START_DEFINE_EXCEPTION(BasisSelectorUnsupportedCellTypeException, BasisSelectorBaseException)
START_DEFINE_EXCEPTION(BasisSelectorUnsupportedPolynomOrderException, BasisSelectorBaseException)
// ************************************************************************************* //

// ****************************** Cubature exceptions ****************************** //
START_DEFINE_EXCEPTION(CubatureBaseException, FEMBaseException)
START_DEFINE_EXCEPTION(CubatureInitializingCubatureException, CubatureBaseException)
START_DEFINE_EXCEPTION(CubatureUnknownException, FEMUnknownException)
// ************************************************************************************* //

// ****************************** Matrix exceptions ************************************ //
START_DEFINE_EXCEPTION(MatrixManagerBaseException, FEMBaseException)
START_DEFINE_EXCEPTION(MatrixManagerEntriesEmptyException, FEMInvalidArgumentErrorException)
START_DEFINE_EXCEPTION(MatrixManagerEntriesNegativeIndicesException, FEMOutOfRangeErrorException)
START_DEFINE_EXCEPTION(MatrixManagerEntriesNotFoundException, FEMOutOfRangeErrorException)
// ************************************************************************************* //

// ****************************** Solvers exceptions ************************************ //
START_DEFINE_EXCEPTION(SolversBaseException, FEMBaseException)
START_DEFINE_EXCEPTION(SolversUnknownException, FEMUnknownException)
START_DEFINE_EXCEPTION(SolversSolutionVectorNotInitializedException, FEMLogicErrorException)
START_DEFINE_EXCEPTION(SolversNodeIDOutOfRangeException, FEMOutOfRangeErrorException)
START_DEFINE_EXCEPTION(SolversCalculatingElectricFieldException, FEMLogicErrorException)
START_DEFINE_EXCEPTION(SolversTimeNegativeException, FEMInvalidArgumentErrorException)
START_DEFINE_EXCEPTION(SolversWritingElectricPotentialsToPosFileException, SolversBaseException)
START_DEFINE_EXCEPTION(SolversWritingElectricFieldVectorsToPosFileException, SolversBaseException)
START_DEFINE_EXCEPTION(SolversSettingSolverParametersException, FEMInvalidArgumentErrorException)
START_DEFINE_EXCEPTION(SolversFileDoesNotExistException, FEMInvalidArgumentErrorException)
START_DEFINE_EXCEPTION(SolversFileIsNotJSONException, FEMInvalidArgumentErrorException)
START_DEFINE_EXCEPTION(SolversUnableToOpenFileException, FEMInvalidArgumentErrorException)
START_DEFINE_EXCEPTION(SolversUnsupportedSolverNameException, FEMInvalidArgumentErrorException)
START_DEFINE_EXCEPTION(SolversFailedToSolveEquationException, SolversBaseException)
START_DEFINE_EXCEPTION(SolversFailedToSetUpLinearProblemException, SolversBaseException)
START_DEFINE_EXCEPTION(SolversFailedToConvergeException, SolversBaseException)

START_DEFINE_JSON_EXCEPTION(SolversFailedToParseJSONFileException, json::parse_error)
START_DEFINE_JSON_EXCEPTION(SolversTypeJSONFileException, json::type_error)
// ************************************************************************************* //

// ****************************** FEMCheckers exceptions ****************************** //
START_DEFINE_EXCEPTION(FEMCheckersBaseException, FEMBaseException)
START_DEFINE_EXCEPTION(FEMCheckersUnsupportedCellTypeException, FEMCheckersBaseException)
START_DEFINE_EXCEPTION(FEMCheckersUnderflowDesiredAccuracyException, FEMUnderflowErrorException)
START_DEFINE_EXCEPTION(FEMCheckersOverflowDesiredAccuracyException, FEMOverflowErrorException)
START_DEFINE_EXCEPTION(FEMCheckersUnsupportedDesiredAccuracyException, FEMInvalidArgumentErrorException)
START_DEFINE_EXCEPTION(FEMCheckersUnderflowPolynomOrderException, FEMUnderflowErrorException)
START_DEFINE_EXCEPTION(FEMCheckersUnsupportedPolynomOrderException, FEMCheckersBaseException)
START_DEFINE_EXCEPTION(FEMCheckersOverflowPolynomOrderException, FEMOverflowErrorException)
START_DEFINE_EXCEPTION(FEMCheckersIndexOutOfRangeException, FEMOutOfRangeErrorException)
START_DEFINE_EXCEPTION(FEMCheckersMatrixMismatchException, FEMCheckersBaseException)
START_DEFINE_EXCEPTION(FEMCheckersBoundaryConditionOutOfRangeException, FEMOutOfRangeErrorException)
START_DEFINE_EXCEPTION(FEMCheckersUnknownException, FEMUnknownException)
// *************************************************************************************//

// ****************************** FEMPrinter exceptions ****************************** //
START_DEFINE_EXCEPTION(FEMPrinterBaseException, FEMBaseException)
START_DEFINE_EXCEPTION(FEMUtilsPrintGraphException, FEMPrinterBaseException)
START_DEFINE_EXCEPTION(FEMUtilsPrintMatrixException, FEMPrinterBaseException)
START_DEFINE_EXCEPTION(FEMPrinterUnknownException, FEMUnknownException)
// *********************************************************************************** //

#endif // !FEM_EXCEPTIONS_HPP
