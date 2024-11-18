# src/FiniteElementMethod/Cubature/CMakeLists.txt

set(CUBATURE_SOURCES
    CubatureManager.cpp
)

set(LIB_FEM_CUBATURE FiniteElementMethod_Cubature)
add_library(${LIB_FEM_CUBATURE} ${CUBATURE_SOURCES})

target_include_directories(${LIB_FEM_CUBATURE} PRIVATE 
    ${Trilinos_INCLUDE_DIRS}
)

if (GPU_CUDA_ON)
    target_include_directories(${LIB_FEM_CUBATURE} PRIVATE ${CUDA_INCLUDE_DIR})
    set_target_properties(${LIB_FEM_CUBATURE} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
