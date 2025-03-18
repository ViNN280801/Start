#ifndef UTILS_EXCEPTIONS_HPP
#define UTILS_EXCEPTIONS_HPP

#include "Utilities/ExceptionMacros.hpp"

// ****************************** Base exceptions ************************************** //
START_DEFINE_EXCEPTION(UtilsBaseException, std::runtime_error)
START_DEFINE_EXCEPTION(UtilsInvalidArgumentException, std::invalid_argument)
START_DEFINE_EXCEPTION(UtilsOutOfRangeException, std::out_of_range)
START_DEFINE_EXCEPTION(UtilsUnknownException, UtilsBaseException)
// ************************************************************************************* //

// ****************************** Utils exceptions ************************************** //
START_DEFINE_EXCEPTION(UtilsMissingRequiredParameterException, UtilsInvalidArgumentException)
START_DEFINE_EXCEPTION(UtilsFailedToOpenConfigurationFileException, UtilsBaseException)
START_DEFINE_EXCEPTION(UtilsFailedToParseConfigurationFileException, UtilsBaseException)
START_DEFINE_EXCEPTION(UtilsNumThreadsOutOfRangeException, UtilsBaseException)
START_DEFINE_EXCEPTION(UtilsGettingConfigDataException, UtilsBaseException)
START_DEFINE_EXCEPTION(UtilsDuplicateNodeValuesException, UtilsBaseException)
START_DEFINE_EXCEPTION(UtilsNodeIDOutOfRangeException, UtilsOutOfRangeException)
START_DEFINE_EXCEPTION(UtilsInvalidNodeIDException, UtilsInvalidArgumentException)
START_DEFINE_EXCEPTION(UtilsInvalidValueForNodeIDsException, UtilsInvalidArgumentException)

START_DEFINE_EXCEPTION(UtilsFailedToOpenFileException, UtilsBaseException)
START_DEFINE_EXCEPTION(UtilsInvalidJSONFileException, UtilsBaseException)
// ************************************************************************************* //

// ****************************** Gmsh utils exceptions ************************************** //
START_DEFINE_EXCEPTION(GmshUtilsBaseException, UtilsBaseException)
START_DEFINE_EXCEPTION(GmshUtilsGmshNotInitializedException, GmshUtilsBaseException)
START_DEFINE_EXCEPTION(GmshUtilsFileDoesNotExistException, GmshUtilsBaseException)
START_DEFINE_EXCEPTION(GmshUtilsFileIsDirectoryException, GmshUtilsBaseException)
START_DEFINE_EXCEPTION(GmshUtilsFileExtensionIsNotMshException, GmshUtilsBaseException)
START_DEFINE_EXCEPTION(GmshUtilsFileIsEmptyException, GmshUtilsBaseException)
START_DEFINE_EXCEPTION(GmshUtilsNoVolumeEntitiesException, GmshUtilsBaseException)
START_DEFINE_EXCEPTION(GmshUtilsNoBoundaryTagsException, GmshUtilsBaseException)
START_DEFINE_EXCEPTION(GmshUtilsNoNodeTagsException, GmshUtilsBaseException)
START_DEFINE_EXCEPTION(GmshUtilsNoTriangleCellsException, GmshUtilsBaseException)
START_DEFINE_EXCEPTION(GmshUtilsFailedToFillTriangleNodeTagsMapException, GmshUtilsBaseException)
START_DEFINE_EXCEPTION(GmshUtilsFailedToFillTriangleCentersMapException, GmshUtilsBaseException)
START_DEFINE_EXCEPTION(GmshUtilsFailedToFillCellsMapException, GmshUtilsBaseException)
START_DEFINE_EXCEPTION(GmshUtilsNoPhysicalGroupsException, GmshUtilsBaseException)
START_DEFINE_EXCEPTION(GmshUtilsPhysicalGroupNotFoundException, GmshUtilsBaseException)
START_DEFINE_EXCEPTION(GmshUtilsNoSurfaceCoordsException, GmshUtilsBaseException)
START_DEFINE_EXCEPTION(GmshUtilsInvalidSurfaceCoordsException, GmshUtilsBaseException)
START_DEFINE_EXCEPTION(GmshUtilsFailedToFillTriangleCellsMapException, GmshUtilsBaseException)
START_DEFINE_EXCEPTION(GmshUtilsFailedToCreateTriangleException, GmshUtilsBaseException)
// ************************************************************************************* //

#endif // !UTILS_EXCEPTIONS_HPP
