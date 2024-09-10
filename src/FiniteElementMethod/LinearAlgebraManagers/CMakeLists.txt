# src/FiniteElementMethod/LinearAlgebraManagers

set(LINEAR_ALGEBRA_MANAGERS_SOURCES
    MatrixManager.cpp
    VectorManager.cpp
)

add_library(FiniteElementMethod_LinearAlgebraManagers ${LINEAR_ALGEBRA_MANAGERS_SOURCES})

target_include_directories(FiniteElementMethod_LinearAlgebraManagers PRIVATE 
    ${Trilinos_INCLUDE_DIRS}
)
