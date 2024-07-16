set(PIC_SOURCES
    Grid3D.cpp
    ParticleTracker.cpp
)

add_library(ParticleTracker ${PIC_SOURCES})

target_link_libraries(ParticleTracker PRIVATE
    ${GMSH_LIBRARIES}
    ${CGAL_LIBRARIES}
    ${Boost_LIBRARIES}
    ${MPI_LIBRARIES}
    ${Trilinos_LIBRARIES}
)

target_include_directories(ParticleTracker PRIVATE
    ${CGAL_INCLUDE_DIRS}
    ${GMSH_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${MPI_INCLUDE_PATH}
    ${Trilinos_INCLUDE_DIRS})
