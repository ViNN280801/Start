# src/FiniteElementMethod/LinearAlgebraManagers

set(LINEAR_ALGEBRA_MANAGERS_SOURCES
    MatrixManager.cpp
    VectorManager.cpp
)

set(LIB_FEM_LINEAR_ALGEBRA_MANAGERS FiniteElementMethod_LinearAlgebraManagers)
add_library(${LIB_FEM_LINEAR_ALGEBRA_MANAGERS} ${LINEAR_ALGEBRA_MANAGERS_SOURCES})

target_include_directories(${LIB_FEM_LINEAR_ALGEBRA_MANAGERS} PRIVATE 
    ${Trilinos_INCLUDE_DIRS}
)

if (GPU_CUDA_ON)
    target_include_directories(${LIB_FEM_LINEAR_ALGEBRA_MANAGERS} PRIVATE ${CUDA_INCLUDE_DIR})
    set_target_properties(${LIB_FEM_LINEAR_ALGEBRA_MANAGERS} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
