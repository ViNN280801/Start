# src/FiniteElementMethod/Cell

set(CELL_SOURCES
    CellSelector.cpp
    CellSelectorException.cpp
)

set(LIB_FEM_CELL FiniteElementMethod_Cell)
add_library(${LIB_FEM_CELL} ${CELL_SOURCES})

target_include_directories(${LIB_FEM_CELL} PRIVATE 
    ${Trilinos_INCLUDE_DIRS}
)

if (GPU_CUDA_ON)
    target_include_directories(${LIB_FEM_CELL} PRIVATE ${CUDA_INCLUDE_DIR})
    set_target_properties(${LIB_FEM_CELL} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
