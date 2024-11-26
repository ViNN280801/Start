# src/FiniteElementMethod/CMakeLists.txt

set(FEM_SOURCES
    FEMCheckers.cpp
    FEMInitializer.cpp
    FEMPrinter.cpp
    GSMAssembler.cpp
    MatrixEquationSolver.cpp
)

add_compile_options("-flarge-source-files" "-Wmisleading-indentation")

add_subdirectory(BoundaryConditions)
add_subdirectory(Cell)
add_subdirectory(Cubature)
add_subdirectory(LinearAlgebraManagers)

set(LIB_FEM FiniteElementMethod)
add_library(${LIB_FEM} ${FEM_SOURCES})

find_package(TBB REQUIRED)
target_link_libraries(${LIB_FEM} PRIVATE
    ${LIB_FEM_BOUNDARY_CONDITIONS}
    ${LIB_FEM_CELL}
    ${LIB_FEM_CUBATURE}
    ${LIB_FEM_LINEAR_ALGEBRA_MANAGERS}

    ${Trilinos_LIBRARIES}
    TBB::tbb
)

target_include_directories(${LIB_FEM} PRIVATE
    ${Trilinos_INCLUDE_DIRS}
)
