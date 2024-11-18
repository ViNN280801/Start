# src/SessionManagement/CMakeLists.txt

set(SESSION_MANAGEMENT_SOURCES
    GmshSessionManager.cpp
)

set(LIB_SESSION_MANAGERS SessionManagers)
add_library(${LIB_SESSION_MANAGERS} ${SESSION_MANAGEMENT_SOURCES})

target_link_libraries(${LIB_SESSION_MANAGERS} PRIVATE
    ${GMSH_LIBRARIES}
)

target_include_directories(${LIB_SESSION_MANAGERS} PRIVATE
    ${GMSH_INC}
)
