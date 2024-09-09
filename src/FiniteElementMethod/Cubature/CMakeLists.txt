# src/FiniteElementMethod/Cubature

set(CUBATURE_SOURCES
    CubatureManager.cpp
)

add_library(FiniteElementMethod_Cubature ${CUBATURE_SOURCES})

target_include_directories(FiniteElementMethod_Cubature PRIVATE 
    ${Trilinos_INCLUDE_DIRS}
)
