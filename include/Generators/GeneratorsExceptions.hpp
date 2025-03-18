#ifndef GENERATORS_EXCEPTIONS_HPP
#define GENERATORS_EXCEPTIONS_HPP

#include "Utilities/ExceptionMacros.hpp"

// ****************************** Base exceptions ************************************** //
START_DEFINE_EXCEPTION(GeneratorsBaseException, std::runtime_error)
START_DEFINE_EXCEPTION(GeneratorsInvalidArgumentException, std::invalid_argument)
START_DEFINE_EXCEPTION(GeneratorsUnknownException, GeneratorsBaseException)
// ************************************************************************************* //

// ****************************** Particle generators exceptions ****************************** //
START_DEFINE_EXCEPTION(ParticleGeneratorsBaseException, GeneratorsBaseException)
START_DEFINE_EXCEPTION(ParticleGeneratorsUnknownException, ParticleGeneratorsBaseException)
START_DEFINE_EXCEPTION(ParticleGeneratorsZeroCountException, GeneratorsInvalidArgumentException)
START_DEFINE_EXCEPTION(ParticleGeneratorsInvalidCoordinateException, GeneratorsInvalidArgumentException)
START_DEFINE_EXCEPTION(ParticleGeneratorsInvalidNumberOfCoordinatesException, GeneratorsInvalidArgumentException)
START_DEFINE_EXCEPTION(ParticleGeneratorsUnknownParticleTypeException, GeneratorsInvalidArgumentException)
START_DEFINE_EXCEPTION(ParticleGeneratorsZeroEnergyException, GeneratorsInvalidArgumentException)
START_DEFINE_EXCEPTION(ParticleGeneratorsEmptyBaseCoordinatesException, GeneratorsInvalidArgumentException)
START_DEFINE_EXCEPTION(ParticleGeneratorsZeroNormalVectorMagnitudeException, GeneratorsInvalidArgumentException)
START_DEFINE_EXCEPTION(ParticleGeneratorsInvalidNormalVectorSizeException, GeneratorsInvalidArgumentException)
// ******************************************************************************************** //

// ****************************** Real number generators exceptions ****************************** //
START_DEFINE_EXCEPTION(RealNumberGeneratorsBaseException, GeneratorsBaseException)
START_DEFINE_EXCEPTION(RealNumberGeneratorsUnknownException, RealNumberGeneratorsBaseException)
START_DEFINE_EXCEPTION(RealNumberGeneratorsZeroCountException, GeneratorsInvalidArgumentException)
START_DEFINE_EXCEPTION(RealNumberGeneratorsGenerateSequenceException, GeneratorsInvalidArgumentException)
// ************************************************************************************************ //

// ============================= CUDA particle generators exceptions ============================= //
START_DEFINE_EXCEPTION(ParticleGeneratorsDeviceBaseException, GeneratorsBaseException)
START_DEFINE_EXCEPTION(ParticleGeneratorsDeviceUnknownException, ParticleGeneratorsDeviceBaseException)
START_DEFINE_EXCEPTION(ParticleGeneratorsDeviceZeroCountException, GeneratorsInvalidArgumentException)
START_DEFINE_EXCEPTION(ParticleGeneratorsDeviceGenerateSequenceException, GeneratorsInvalidArgumentException)
START_DEFINE_EXCEPTION(ParticleGeneratorsDeviceUnknownParticleTypeException, GeneratorsInvalidArgumentException)
// ================================================================================================ //

// ============================= CUDA real number generators exceptions ============================= //
START_DEFINE_EXCEPTION(RealNumberGeneratorsDeviceBaseException, GeneratorsBaseException)
START_DEFINE_EXCEPTION(RealNumberGeneratorsDeviceUnknownException, RealNumberGeneratorsDeviceBaseException)
START_DEFINE_EXCEPTION(RealNumberGeneratorsDeviceGenerateSequenceException, RealNumberGeneratorsDeviceBaseException)
// ================================================================================================ //

#endif // !GENERATORS_EXCEPTIONS_HPP
