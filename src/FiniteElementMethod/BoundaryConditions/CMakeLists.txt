set(BOUNDARY_CONDITIONS_SOURCES
    VectorBoundaryConditionsManager.cpp
    MatrixBoundaryConditionsManager.cpp
    BoundaryConditionsManager.cpp
)

add_library(FiniteElementMethod_BoundaryConditions ${BOUNDARY_CONDITIONS_SOURCES})

target_include_directories(FiniteElementMethod_BoundaryConditions PRIVATE 
    ${Trilinos_INCLUDE_DIRS}
)