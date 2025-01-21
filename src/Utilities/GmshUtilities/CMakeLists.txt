# src/Utilities/GmshUtilities/CMakeLists.txt

set(GMSH_UTILITIES_SOURCES
    GmshMesher.cpp
    GmshUtils.cpp
)

set(LIB_GMSH_UTILITIES GmshUtilities)
add_library(${LIB_GMSH_UTILITIES} ${GMSH_UTILITIES_SOURCES})

target_link_libraries(${LIB_GMSH_UTILITIES} PRIVATE ${GMSH_LIBRARIES})
target_include_directories(${LIB_GMSH_UTILITIES} PRIVATE ${GMSH_INC})
