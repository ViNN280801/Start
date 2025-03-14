#ifndef ASSEMBLERS_EXCEPTIONS_HPP
#define ASSEMBLERS_EXCEPTIONS_HPP

#include "Utilities/ExceptionMacros.hpp"

// ****************************** Base exceptions ************************************** //
START_DEFINE_EXCEPTION(AssemblersBaseException, std::runtime_error)
START_DEFINE_EXCEPTION(AssemblersUnknownException, AssemblersBaseException)
// ************************************************************************************* //

// ****************************** GSM assembler exceptions ****************************** //
START_DEFINE_EXCEPTION(GSMAssemblerBaseException, AssemblersBaseException)
START_DEFINE_EXCEPTION(GSMAssemblerGettingMatrixEntriesException, std::logic_error)
START_DEFINE_EXCEPTION(GSMAssemblerComputingLocalStiffnessMatricesException, std::logic_error)
START_DEFINE_EXCEPTION(GSMAssemblerUnknownException, AssemblersUnknownException)
// ************************************************************************************* //

#endif // !ASSEMBLERS_EXCEPTIONS_HPP
