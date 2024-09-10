# src/FiniteElementMethod

set(FEM_SOURCES
    FEMCheckers.cpp
    FEMPrinter.cpp
    GSMAssemblier.cpp
    MatrixEquationSolver.cpp
)

if (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flarge-source-files -Wmisleading-indentation")
endif()

add_subdirectory(BoundaryConditions)
add_subdirectory(Cell)
add_subdirectory(Cubature)
add_subdirectory(LinearAlgebraManagers)

add_library(FiniteElementMethod ${FEM_SOURCES})

target_link_libraries(FiniteElementMethod PRIVATE
    FiniteElementMethod_BoundaryConditions
    FiniteElementMethod_Cell
    FiniteElementMethod_Cubature
    FiniteElementMethod_LinearAlgebraManagers

    ${Trilinos_LIBRARIES}
)

target_include_directories(FiniteElementMethod PRIVATE
    ${Trilinos_INCLUDE_DIRS}
)
