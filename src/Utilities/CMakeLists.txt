# src/Utilities/CMakeLists.txt

set(GMSH_UTILITIES_DIR GmshUtilities)
set(UTILITIES_SOURCES
    ${GMSH_UTILITIES_DIR}/GmshMesher.cpp
    ${GMSH_UTILITIES_DIR}/GmshUtils.cpp

    ConfigParser.cpp
    Timer.cpp
    Utilities.cpp
)

set(LIB_UTILITIES Utilities)
add_library(${LIB_UTILITIES} ${UTILITIES_SOURCES})
