set(FEM_SOURCES
    GSMatrixAssemblier.cpp
    MatrixEquationSolver.cpp
    SolutionVector.cpp
)

if (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
endif()

add_library(FiniteElementMethod ${FEM_SOURCES})

target_link_libraries(FiniteElementMethod PRIVATE
    ${Trilinos_LIBRARIES}
)

target_include_directories(FiniteElementMethod PRIVATE
    ${Trilinos_INCLUDE_DIRS})
