# src/FiniteElementMethod/BoundaryConditions

set(BOUNDARY_CONDITIONS_SOURCES
    VectorBoundaryConditionsManager.cpp
    MatrixBoundaryConditionsManager.cpp
    BoundaryConditionsManager.cpp
)

set(LIB_NAME FiniteElementMethod_BoundaryConditions)
add_library(${LIB_NAME} ${BOUNDARY_CONDITIONS_SOURCES})

target_include_directories(${LIB_NAME} PRIVATE 
    ${Trilinos_INCLUDE_DIRS}
)

if (GPU_CUDA_ON)
    target_include_directories(${LIB_NAME} PRIVATE ${CUDA_INCLUDE_DIR})
    set_target_properties(${LIB_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
