# src/FiniteElementMethod/BoundaryConditions

set(BOUNDARY_CONDITIONS_SOURCES
    VectorBoundaryConditionsManager.cpp
    MatrixBoundaryConditionsManager.cpp
    BoundaryConditionsManager.cpp
)

set(LIB_FEM_BOUNDARY_CONDITIONS FiniteElementMethod_BoundaryConditions)
add_library(${LIB_FEM_BOUNDARY_CONDITIONS} ${BOUNDARY_CONDITIONS_SOURCES})

target_include_directories(${LIB_FEM_BOUNDARY_CONDITIONS} PRIVATE 
    ${Trilinos_INCLUDE_DIRS}
)

if (GPU_CUDA_ON)
    target_include_directories(${LIB_FEM_BOUNDARY_CONDITIONS} PRIVATE ${CUDA_INCLUDE_DIR})
    set_target_properties(${LIB_FEM_BOUNDARY_CONDITIONS} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
