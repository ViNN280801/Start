set(SESSION_MANAGEMENT_SOURCES
    GmshSessionManager.cpp
)

add_library(SessionManagers ${SESSION_MANAGEMENT_SOURCES})

target_link_libraries(SessionManagers PRIVATE
    ${GMSH_LIBRARIES}
)

target_include_directories(SessionManagers PRIVATE
    ${GMSH_INC}
)
