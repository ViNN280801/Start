# src/FiniteElementMethod/Cubature

set(CUBATURE_SOURCES
    CubatureManager.cpp
)

set(LIB_NAME FiniteElementMethod_Cubature)
add_library(${LIB_NAME} ${CUBATURE_SOURCES})

target_include_directories(${LIB_NAME} PRIVATE 
    ${Trilinos_INCLUDE_DIRS}
)

if (GPU_CUDA_ON)
    target_include_directories(${LIB_NAME} PRIVATE ${CUDA_INCLUDE_DIR})
    set_target_properties(${LIB_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
