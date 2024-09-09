# Root CMakeLists.txt

cmake_minimum_required(VERSION 3.14)
project(Tests)

################# Adding third-party libs #################
find_package(Threads REQUIRED)

# Gtests
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/release-1.10.0.tar.gz
)
FetchContent_MakeAvailable(googletest)

# HDF5
find_package(HDF5 REQUIRED COMPONENTS CXX)
if(HDF5_FOUND)
    message(STATUS "HDF5 library found: ${HDF5_LIBRARIES}")
else()
    message(FATAL_ERROR "HDF5 library not found. Please ensure HDF5 is installed.")
endif()
include_directories(${HDF5_INCLUDE_DIRS})

# GMSH
if (WIN32)
    find_library(GMSH_LIBRARIES
                NAMES gmsh
                PATHS "C:/Program Files/gmsh-4.12.2-Windows64-sdk/lib"
                NO_DEFAULT_PATH)
else()
    find_library(GMSH_LIBRARIES
                NAMES gmsh
                PATHS /usr/lib 
                      /usr/local/lib
                      /usr/lib/x86_64-linux-gnu
                      /usr/lib64 
                      /usr/lib64/mpich/lib 
                      /usr/lib64/openmpi/lib
                NO_DEFAULT_PATH)
endif()

if(GMSH_LIBRARIES)
    message(STATUS "GMSH library found: ${GMSH_LIBRARIES}")
else()
    message(FATAL_ERROR "GMSH library not found. Please ensure GMSH is installed.")
endif()

# GMP
if(MSVC)
    find_package(GMP REQUIRED)
    if(GMP_FOUND)
        message(STATUS "GMP library found: ${GMP_LIBRARIES}")
    else()
        message(FATAL_ERROR "GMP library not found.")
    endif()
else()
    set(GMP_LIBRARIES gmp)
    message(STATUS "GMP libraries set to default: ${GMP_LIBRARIES}")
endif()
include_directories(${GMP_INCLUDE_DIR})

# TBB
find_package(TBB QUIET)
if (TBB_FOUND)
    set(TBB_NAMESPACE TBB::tbb)
    message(STATUS "TBB library found: ${TBB_LIBRARIES}")
else()
    message(WARNING "TBB library not found. Proceeding without TBB.")
endif()
include_directories(${TBB_INCLUDE_DIRS})

# Boost
find_package(Boost REQUIRED)
if(Boost_FOUND)
    message(STATUS "Boost library found: ${Boost_LIBRARIES}")
else()
    message(FATAL_ERROR "Boost library not found. Please ensure Boost is installed.")
endif()
include_directories(${Boost_INCLUDE_DIRS})

# MPI (for parallelization)
if (WIN32)
    set(MPIEXEC_EXECUTABLE "C:/Program Files/Microsoft MPI/Bin/mpiexec.exe")
    set(MPI_CXX_HEADER_DIR "C:/Program Files (x86)/Microsoft SDKs/MPI/../include")
    set(MPI_CXX_LIB_NAMES "msmpi")
    set(MPI_msmpi_LIBRARY "C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64/msmpi.lib")
    set(MPI_CXX_LIBRARIES "C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64/msmpi.lib")
    set(MPI_CXX_COMPILER "cl.exe")
    include_directories(${MPI_CXX_HEADER_DIR})
    link_directories(${MPI_CXX_LIBRARIES})
else()
    find_package(MPI REQUIRED)
    if(MPI_FOUND)
        message(STATUS "MPI libraries found: ${MPI_LIBRARIES}")
    else()
        message(FATAL_ERROR "MPI libraries not found. Please ensure MPI is installed.")
    endif()
endif()
include_directories(${MPI_INCLUDE_PATH})

# Trilinos
if(NOT "$ENV{TRILINOS_DIR}" STREQUAL "")
    set(TRILINOS_DIR "$ENV{TRILINOS_DIR}")
    find_package(Trilinos REQUIRED PATHS "${TRILINOS_DIR}")
    message(STATUS "Trilinos directory set to custom environment variable")
else()
    find_package(Trilinos REQUIRED PATHS "/usr/local/lib/cmake/Trilinos")
    if(Trilinos_FOUND)
        message(STATUS "Trilinos library found: ${Trilinos_LIBRARIES}")
    else()
        message(FATAL_ERROR "Trilinos library not found. Please ensure Trilinos is installed.")
    endif()
endif()
include_directories(${Trilinos_INCLUDE_DIRS})
link_directories(${Trilinos_LIBRARY_DIRS})
#####################################################################################

################# Setting up build types #################
# Build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG)
    message(STATUS "*** Compiling in DEBUG mode ***")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_definitions(RELEASE)
    message(STATUS "*** Compiling in RELEASE mode ***")
else()
    message(WARNING "Unknown build type: ${CMAKE_BUILD_TYPE}")
endif()

# Custom Start build modes
option(START_DEBUG "Compile the project in Start Debug mode" OFF)
option(START_RELEASE "Compile the project in Start Release mode" OFF)
if(START_DEBUG)
    add_compile_definitions(START_DEBUG)
    message(STATUS "*** Compiling in START_DEBUG mode ***")
elseif(START_RELEASE)
    add_compile_definitions(START_RELEASE)
    message(STATUS "*** Compiling in START_RELEASE mode ***")
else()
    message(WARNING "Neither START_DEBUG nor START_RELEASE is defined.")
endif()
#####################################################################################

set(TARGET_NAME all_tests)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -O3 -Wall -Wextra -Werror -Wno-maybe-uninitialized -Wmisleading-indentation -flarge-source-files")

set(DATA_HANDLING_SRC_PATH ${CMAKE_SOURCE_DIR}/../src/DataHandling)
set(DATA_HANDLING_INC_PATH ${CMAKE_SOURCE_DIR}/../include/DataHandling)
set(FINITE_ELEMENT_METHOD_SRC_PATH 
    ${CMAKE_SOURCE_DIR}/../src/FiniteElementMethod
)
set(FINITE_ELEMENT_METHOD_INC_PATH
    ${CMAKE_SOURCE_DIR}/../include/FiniteElementMethod
    ${CMAKE_SOURCE_DIR}/../include/FiniteElementMethod/BoundaryConditions
    ${CMAKE_SOURCE_DIR}/../include/FiniteElementMethod/Cell
    ${CMAKE_SOURCE_DIR}/../include/FiniteElementMethod/Cubature
)
set(GENERATORS_SRC_PATH ${CMAKE_SOURCE_DIR}/../src/Generators)
set(GENERATORS_INC_PATH ${CMAKE_SOURCE_DIR}/../include/Generators)
set(GEOMETRY_SRC_PATH ${CMAKE_SOURCE_DIR}/../src/Geometry)
set(GEOMETRY_INC_PATH ${CMAKE_SOURCE_DIR}/../include/Geometry)
set(PARTICLES_SRC_PATH ${CMAKE_SOURCE_DIR}/../src/Particles)
set(PARTICLES_INC_PATH ${CMAKE_SOURCE_DIR}/../include/Particles)
set(UTILITIES_SRC_PATH ${CMAKE_SOURCE_DIR}/../src/Utilities)
set(UTILITIES_INC_PATH ${CMAKE_SOURCE_DIR}/../include/Utilities)

set(DATA_HANDLING_SOURCES
    ${DATA_HANDLING_SRC_PATH}/HDF5Handler.cpp
    ${DATA_HANDLING_SRC_PATH}/TetrahedronMeshManager.cpp
)
set(FINITE_ELEMENT_METHOD_SOURCES
    ${FINITE_ELEMENT_METHOD_SRC_PATH}/BoundaryConditions/BoundaryConditionsManager.cpp
    ${FINITE_ELEMENT_METHOD_SRC_PATH}/BoundaryConditions/MatrixBoundaryConditionsManager.cpp
    ${FINITE_ELEMENT_METHOD_SRC_PATH}/BoundaryConditions/VectorBoundaryConditionsManager.cpp

    ${FINITE_ELEMENT_METHOD_SRC_PATH}/Cell/CellSelector.cpp
    ${FINITE_ELEMENT_METHOD_SRC_PATH}/Cell/CellSelectorException.cpp

    ${FINITE_ELEMENT_METHOD_SRC_PATH}/FEMCheckers.cpp
    ${FINITE_ELEMENT_METHOD_SRC_PATH}/FEMPrinter.cpp
    ${FINITE_ELEMENT_METHOD_SRC_PATH}/GSMatrixAssemblier.cpp
    ${FINITE_ELEMENT_METHOD_SRC_PATH}/MatrixEquationSolver.cpp
    ${FINITE_ELEMENT_METHOD_SRC_PATH}/SolutionVector.cpp
)
set(GENERATORS_SOURCES
    ${GENERATORS_SRC_PATH}/RealNumberGenerator.cpp
    ${GENERATORS_SRC_PATH}/VolumeCreator.cpp
)
set(GEOMETRY_SOURCES
    ${GEOMETRY_SRC_PATH}/MathVector.cpp
    ${GEOMETRY_SRC_PATH}/Mesh.cpp
    ${GEOMETRY_SRC_PATH}/RayTriangleIntersection.cpp
)
set(PARTICLES_SOURCES
    ${PARTICLES_SRC_PATH}/Particle.cpp
)
set(UTILITIES_SOURCES
    ${UTILITIES_SRC_PATH}/Utilities.cpp
    ${UTILITIES_SRC_PATH}/ConfigParser.cpp
)
set(SOURCES
    ${DATA_HANDLING_SOURCES}
    ${FINITE_ELEMENT_METHOD_SOURCES}
    ${GENERATORS_SOURCES}
    ${GEOMETRY_SOURCES}
    ${PARTICLES_SOURCES}
    ${UTILITIES_SOURCES}
)

set(DATA_HANDLING_TEST_SOURCES
    DataHandling/HDF5HandlerTests.cpp
    DataHandling/TetrahedronMeshManagerTests.cpp
)
set(FINITE_ELEMENT_METHOD_TEST_SOURCES
    FiniteElementMethod/BoundaryConditions/MatrixBoundaryConditionsTests.cpp
    FiniteElementMethod/BoundaryConditions/VectorBoundaryConditionsTests.cpp

    FiniteElementMethod/Cell/CellSelectorTests.cpp

    FiniteElementMethod/Cubature/BasisSelectorTests.cpp
)
set(GENERATORS_TEST_SOURCES
    Generators/RealNumberGeneratorTests.cpp
    Generators/VolumeCreatorTests.cpp
)
set(GEOMETRY_TEST_SOURCES
    Geometry/MathVectorTests.cpp
    Geometry/MeshTests.cpp
    Geometry/RayTriangleIntersectionTests.cpp
)
set(PARTICLES_TEST_SOURCES
    Particles/ParticleTests.cpp
)
set(UTILITIES_TEST_SOURCES
    Utilities/ConfigParserTests.cpp
)
set(TEST_SOURCES
    ${DATA_HANDLING_TEST_SOURCES}
    ${FINITE_ELEMENT_METHOD_TEST_SOURCES}
    ${GENERATORS_TEST_SOURCES}
    ${GEOMETRY_TEST_SOURCES}
    ${PARTICLES_TEST_SOURCES}
    ${UTILITIES_TEST_SOURCES}
)

set(INCLUDE_DIRECTORIES
    ${INCLUDE_DIRECTORIES}
    ${CMAKE_SOURCE_DIR}/../include
    ${DATA_HANDLING_INC_PATH}
    ${CMAKE_SOURCE_DIR}/../include/FiniteElementMethod
    ${CMAKE_SOURCE_DIR}/../include/FiniteElementMethod/BoundaryConditions
    ${CMAKE_SOURCE_DIR}/../include/FiniteElementMethod/Cell
    ${CMAKE_SOURCE_DIR}/../include/FiniteElementMethod/Cubature
    ${GENERATORS_INC_PATH}
    ${GEOMETRY_INC_PATH}
    ${PARTICLES_INC_PATH}
    ${CMAKE_SOURCE_DIR}/../include/ParticleTracker
    ${UTILITIES_INC_PATH}
)

include_directories(${INCLUDE_DIRECTORIES})

enable_testing()
include(CTest)

set(DATA_HANDLING_TESTS DataHandlingTests)
set(GENERATORS_TESTS GeneratorTests)
set(GEOMETRY_TESTS GeometryTests)
set(PARTICLE_TESTS ParticleTests)
set(UTILITIES_TESTS UtilitiesTests)

add_test(NAME ${DATA_HANDLING_TESTS} COMMAND ${DATA_HANDLING_TESTS})
add_test(NAME ${GENERATORS_TESTS} COMMAND ${GENERATORS_TESTS})
add_test(NAME ${GEOMETRY_TESTS} COMMAND ${GEOMETRY_TESTS})
add_test(NAME ${PARTICLE_TESTS} COMMAND ${PARTICLE_TESTS})
add_test(NAME ${UTILITIES_TESTS} COMMAND ${UTILITIES_TESTS})

add_executable(${TARGET_NAME}
    main.cpp

    ${SOURCES}
    ${TEST_SOURCES}
)

# Disable -Werror for GoogleTest
target_compile_options(gtest PRIVATE -Wno-error)
target_compile_options(gmock PRIVATE -Wno-error)

target_link_libraries(${TARGET_NAME} PRIVATE
    gtest
    gtest_main
    gmock
    gmock_main
    Threads::Threads
    ${CGAL_NAMESPACE}
    ${CGAL_LIBRARIES}
    ${GMSH_LIBRARIES}
    ${Boost_LIBRARIES}
    ${GMP_LIBRARIES}
    ${HDF5_LIBRARIES}
    ${TBB_NAMESPACE}
    ${TBB_LIBRARIES}
    ${MPI_LIBRARIES}
    ${Trilinos_LIBRARIES}
)

target_include_directories(${TARGET_NAME} PRIVATE ${INCLUDE_DIRECTORIES})

include(CTest)
enable_testing()
