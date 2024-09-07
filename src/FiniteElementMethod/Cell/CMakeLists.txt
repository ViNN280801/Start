# src/FiniteElementMethod/Cell

set(CELL_SOURCES
    CellSelector.cpp
    CellSelectorException.cpp
)

add_library(FiniteElementMethod_Cell ${CELL_SOURCES})

target_include_directories(FiniteElementMethod_Cell PRIVATE 
    ${Trilinos_INCLUDE_DIRS}
)