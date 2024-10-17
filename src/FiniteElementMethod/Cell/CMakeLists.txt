# src/FiniteElementMethod/Cell

set(CELL_SOURCES
    CellSelector.cpp
    CellSelectorException.cpp
)

set(LIB_NAME FiniteElementMethod_Cell)
add_library(${LIB_NAME} ${CELL_SOURCES})

target_include_directories(${LIB_NAME} PRIVATE 
    ${Trilinos_INCLUDE_DIRS}
)

if (GPU_CUDA_ON)
    target_include_directories(${LIB_NAME} PRIVATE ${CUDA_INCLUDE_DIR})
    set_target_properties(${LIB_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
