#ifndef GENERATORS_EXCEPTIONS_HPP
#define GENERATORS_EXCEPTIONS_HPP

#include "Utilities/ExceptionMacros.hpp"

// ****************************** Base exceptions ************************************** //
START_DEFINE_EXCEPTION(GeneratorsBaseException, std::runtime_error)
START_DEFINE_EXCEPTION(GeneratorsUnknownException, GeneratorsBaseException)
// ************************************************************************************* //

// ****************************** Particle generators exceptions ****************************** //
START_DEFINE_EXCEPTION(ParticleGeneratorsBaseException, GeneratorsBaseException)
START_DEFINE_EXCEPTION(ParticleGeneratorsUnknownException, ParticleGeneratorsBaseException)
START_DEFINE_EXCEPTION(ParticleGeneratorsZeroCountException, std::invalid_argument)
START_DEFINE_EXCEPTION(ParticleGeneratorsInvalidCoordinateException, std::invalid_argument)
START_DEFINE_EXCEPTION(ParticleGeneratorsInvalidNumberOfCoordinatesException, std::invalid_argument)
START_DEFINE_EXCEPTION(ParticleGeneratorsUnknownParticleTypeException, std::invalid_argument)
START_DEFINE_EXCEPTION(ParticleGeneratorsZeroEnergyException, std::invalid_argument)
START_DEFINE_EXCEPTION(ParticleGeneratorsEmptyBaseCoordinatesException, std::invalid_argument)
START_DEFINE_EXCEPTION(ParticleGeneratorsZeroNormalVectorMagnitudeException, std::invalid_argument)
START_DEFINE_EXCEPTION(ParticleGeneratorsInvalidNormalVectorSizeException, std::invalid_argument)
// ******************************************************************************************** //

// ****************************** Real number generators exceptions ****************************** //
START_DEFINE_EXCEPTION(RealNumberGeneratorsBaseException, GeneratorsBaseException)
START_DEFINE_EXCEPTION(RealNumberGeneratorsUnknownException, RealNumberGeneratorsBaseException)
START_DEFINE_EXCEPTION(RealNumberGeneratorsZeroCountException, std::invalid_argument)
START_DEFINE_EXCEPTION(RealNumberGeneratorsGenerateSequenceException, std::invalid_argument)
// ************************************************************************************************ //

// ============================= CUDA particle generators exceptions ============================= //
START_DEFINE_EXCEPTION(ParticleGeneratorsDeviceBaseException, GeneratorsBaseException)
START_DEFINE_EXCEPTION(ParticleGeneratorsDeviceUnknownException, ParticleGeneratorsDeviceBaseException)
START_DEFINE_EXCEPTION(ParticleGeneratorsDeviceZeroCountException, std::invalid_argument)
START_DEFINE_EXCEPTION(ParticleGeneratorsDeviceGenerateSequenceException, std::invalid_argument)
START_DEFINE_EXCEPTION(ParticleGeneratorsDeviceUnknownParticleTypeException, std::invalid_argument)
// ================================================================================================ //

// ============================= CUDA real number generators exceptions ============================= //
START_DEFINE_EXCEPTION(RealNumberGeneratorsDeviceBaseException, GeneratorsBaseException)
START_DEFINE_EXCEPTION(RealNumberGeneratorsDeviceUnknownException, RealNumberGeneratorsDeviceBaseException)
START_DEFINE_EXCEPTION(RealNumberGeneratorsDeviceGenerateSequenceException, RealNumberGeneratorsDeviceBaseException)
// ================================================================================================ //

#endif // !GENERATORS_EXCEPTIONS_HPP
