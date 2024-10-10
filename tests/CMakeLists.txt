# tests/CMakeLists.txt

cmake_minimum_required(VERSION 3.14)
project(Tests)

################# Adding third-party libs #################
find_package(Threads REQUIRED)
message(STATUS "⚙️  Threads library found and linked successfully.")

###############################################################
#                     Gtest Setup                             #
###############################################################
include(FetchContent)
message(STATUS "🔍 Fetching GoogleTest library...")

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/release-1.10.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googletest)
message(STATUS "✅ GoogleTest successfully downloaded and made available.")
message(STATUS "⚙️ Configuring GoogleTest for project...")

###############################################################
#                     HDF5 Setup                              #
###############################################################
find_package(HDF5 REQUIRED COMPONENTS CXX)
if(HDF5_FOUND)
    message(STATUS "✅ HDF5 found: Include directories at [${HDF5_INCLUDE_DIRS}], Libraries: [${HDF5_LIBRARIES}]")
else()
    message(FATAL_ERROR "❌ HDF5 not found. Please ensure HDF5 is installed and available.")
endif()
include_directories(${HDF5_INCLUDE_DIRS})

###############################################################
#                     GMSH Setup                              #
###############################################################
if(WIN32)
    find_path(GMSH_INC NAMES gmsh.h
        HINTS $ENV{GMSH_INC}
        PATH_SUFFIXES include)

    if(NOT GMSH_INC)
        message(FATAL_ERROR "❌ GMSH include directory not found. Please ensure GMSH is installed and GMSH_INC is set.")
    else()
        message(STATUS "✅ GMSH include directories found at: [${GMSH_INC}]")
    endif()

    include_directories(${GMSH_INC})

    find_library(GMSH_LIBRARIES NAMES gmsh
        HINTS $ENV{GMSH_LIB} 
        PATH_SUFFIXES lib)

    if(NOT GMSH_LIBRARIES)
        message(FATAL_ERROR "❌ GMSH libraries not found. Please ensure GMSH is installed and GMSH_LIB is set.")
    else()
        message(STATUS "✅ GMSH libraries found: ${GMSH_LIBRARIES}")
    endif()

else()
    find_path(GMSH_INC NAMES gmsh.h
        HINTS /usr/include /usr/local/include)

    find_library(GMSH_LIBRARIES NAMES gmsh
        HINTS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)

    if(NOT GMSH_INC OR NOT GMSH_LIBRARIES)
        message(FATAL_ERROR "❌ GMSH not found on Linux/Unix.")
    else()
        message(STATUS "✅ GMSH found: Include directories at [${GMSH_INC}], Libraries at [${GMSH_LIBRARIES}]")
    endif()

    include_directories(${GMSH_INC})
endif()

###############################################################
#                     GMP Setup                               #
###############################################################
if(MSVC)
    find_package(GMP REQUIRED)
    if(GMP_FOUND)
        message(STATUS "✅ GMP found: Include directories at [${GMP_INCLUDE_DIR}], Libraries: [${GMP_LIBRARIES}]")
    else()
        message(FATAL_ERROR "❌ GMP not found. Please ensure GMP is installed and available.")
    endif()
else()
    find_path(GMP_INCLUDE_DIR gmp.h
        HINTS /usr/include/x86_64-linux-gnu /usr/include/CGAL)

    find_library(GMP_LIBRARY NAMES gmp libgmp
        HINTS /usr/lib/x86_64-linux-gnu)

    if(NOT GMP_INCLUDE_DIR OR NOT GMP_LIBRARY)
        message(FATAL_ERROR "❌ GMP not found. Please ensure GMP is installed.")
    else()
        message(STATUS "✅ GMP found: Include directories at [${GMP_INCLUDE_DIR}], Libraries: [${GMP_LIBRARY}]")
    endif()
endif()
include_directories(${GMP_INCLUDE_DIR})
link_libraries(${GMP_LIBRARY})

###############################################################
#                     TBB Setup                               #
###############################################################
find_package(TBB)
if(TBB_FOUND)
    set(TBB_NAMESPACE TBB::tbb)
    message(STATUS "✅ TBB found: Include directories at [${TBB_INCLUDE_DIRS}], Libraries: [${TBB_LIBRARIES}]")
else()
    message(WARNING "⚠️ TBB not found. Continuing without TBB support.")
endif()
include_directories(${TBB_INCLUDE_DIRS})

###############################################################
#                     Boost Setup                             #
###############################################################
cmake_policy(SET CMP0167 NEW)
find_package(Boost REQUIRED)
if(Boost_FOUND)
    message(STATUS "✅ Boost found: Include directories at [${Boost_INCLUDE_DIRS}], Libraries: [${Boost_LIBRARIES}]")
else()
    message(FATAL_ERROR "❌ Boost not found. Please ensure Boost is installed and available.")
endif()
include_directories(${Boost_INCLUDE_DIRS})

###############################################################
#                     MPI Setup                               #
###############################################################
find_package(MPI REQUIRED)
if(MPI_FOUND)
    message(STATUS "✅ MPI found: Include directories at [${MPI_INCLUDE_PATH}], Libraries: [${MPI_LIBRARIES}]")
else()
    message(FATAL_ERROR "❌ MPI not found. Please ensure MPI is installed and available.")
endif()
include_directories(${MPI_INCLUDE_PATH})
link_directories(${MPI_LIBRARY_DIRS})

###############################################################
#                     Trilinos Setup                          #
###############################################################
if(NOT "$ENV{TRILINOS_DIR}" STREQUAL "")
    set(TRILINOS_DIR "$ENV{TRILINOS_DIR}")
    find_package(Trilinos REQUIRED PATHS "${TRILINOS_DIR}")
    message(STATUS "✅ Trilinos directory set to custom environment variable: [${TRILINOS_DIR}]")
else()
    set(TRILINOS_DIR "/usr/local/lib/cmake/Trilinos")
    find_package(Trilinos REQUIRED PATHS "${TRILINOS_DIR}")
    message(STATUS "✅ Trilinos directory set to default location: [${TRILINOS_DIR}]")
endif()

if(Trilinos_FOUND)
    message(STATUS "✅ Trilinos libraries: [${Trilinos_LIBRARIES}]")
    message(STATUS "✅ Trilinos include directories: [${Trilinos_INCLUDE_DIRS}]")
else()
    message(FATAL_ERROR "❌ Trilinos not found. Please ensure Trilinos is installed.")
endif()
include_directories(${Trilinos_INCLUDE_DIRS})
link_directories(${Trilinos_LIBRARY_DIRS})
###############################################################
#               End of Package Configuration                  #
###############################################################

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║           BUILD TYPE CONFIGURATION                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# ╔═► Check if we're building in Debug or Release mode ◄═════════════════════╗
# ║ Define the appropriate macros (`DEBUG` or `RELEASE`) based on the build  ║
# ║ type (Debug, Release, etc.).                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG)
    message(STATUS "⚙️  Compiling in DEBUG mode")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_definitions(RELEASE)
    message(STATUS "🚀 Compiling in RELEASE mode")
else()
    message(WARNING "⚠️ Unknown build type: ${CMAKE_BUILD_TYPE}")
endif()

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║           START DEBUG / RELEASE OPTIONS                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# ╔═► Start-specific Debug and Release options ◄═════════════════════════════╗
# ║ Allow the user to specify custom "Start Debug" or "Start Release" options║
# ║ for specific behavior during compilation.                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
if(START_DEBUG)
    add_compile_definitions(START_DEBUG)
    message(STATUS "⚙️  Start project compiling mode set to START_DEBUG.")
elseif(START_RELEASE)
    add_compile_definitions(START_RELEASE)
    message(STATUS "🚀 Start project compiling mode set to START_RELEASE.")
else()
    message(FATAL_ERROR "⚠️ Neither START_DEBUG nor START_RELEASE is defined.")
endif()
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║              END OF CONFIGURING OPTIONS                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

set(TARGET_NAME all_tests)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -O3 -Wall -Wextra -Werror -Wno-maybe-uninitialized -Wmisleading-indentation -flarge-source-files")

message(STATUS "⚙️  Using C++${CMAKE_CXX_STANDARD} standard.")
message(STATUS "⚙️  C++ extensions set to OFF, because of Kokkos requirements.")
message(STATUS "⚙️  Compiling with C++ flags: ${CMAKE_CXX_FLAGS}")

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
set(PARTICLES_SRC_PATH ${CMAKE_SOURCE_DIR}/../src/Particle)
set(PARTICLES_INC_PATH ${CMAKE_SOURCE_DIR}/../include/Particle)
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

    ${FINITE_ELEMENT_METHOD_SRC_PATH}/Cubature/CubatureManager.cpp

    ${FINITE_ELEMENT_METHOD_SRC_PATH}/LinearAlgebraManagers/MatrixManager.cpp
    ${FINITE_ELEMENT_METHOD_SRC_PATH}/LinearAlgebraManagers/VectorManager.cpp

    ${FINITE_ELEMENT_METHOD_SRC_PATH}/FEMCheckers.cpp
    ${FINITE_ELEMENT_METHOD_SRC_PATH}/FEMPrinter.cpp
    ${FINITE_ELEMENT_METHOD_SRC_PATH}/GSMAssemblier.cpp
    ${FINITE_ELEMENT_METHOD_SRC_PATH}/MatrixEquationSolver.cpp
)
set(GENERATORS_SOURCES
    ${GENERATORS_SRC_PATH}/RealNumberGenerator.cpp
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

    FiniteElementMethod/Cubature/CubatureManagerTests.cpp
    FiniteElementMethod/Cubature/BasisSelectorTests.cpp

    FiniteElementMethod/LinearAlgebraManagers/MatrixManagerTests.cpp
    FiniteElementMethod/LinearAlgebraManagers/VectorManagerTests.cpp

    FiniteElementMethod/FEMCheckersTests.cpp
)
set(GENERATORS_TEST_SOURCES
    Generators/RealNumberGeneratorTests.cpp
)
set(GEOMETRY_TEST_SOURCES
    Geometry/MathVectorTests.cpp
    Geometry/MeshTests.cpp
    Geometry/RayTriangleIntersectionTests.cpp
)
set(PARTICLES_TEST_SOURCES
    Particle/ParticleTests.cpp
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

set(DATA_HANDLING_TESTS DataHandlingTests)
set(GENERATORS_TESTS GeneratorTests)
set(GEOMETRY_TESTS GeometryTests)
set(PARTICLE_TESTS ParticleTests)
set(UTILITIES_TESTS UtilitiesTests)
set(ALL_TEST_SUITES 
    "
    ${DATA_HANDLING_TESTS} 
    ${GENERATORS_TESTS}
    ${GEOMETRY_TESTS}
    ${PARTICLE_TESTS}
    ${UTILITIES_TESTS}
")

add_test(NAME ${DATA_HANDLING_TESTS} COMMAND ${DATA_HANDLING_TESTS})
message(STATUS "🛠️ Added Data Handling Tests: ${DATA_HANDLING_TESTS}")

add_test(NAME ${GENERATORS_TESTS} COMMAND ${GENERATORS_TESTS})
message(STATUS "🛠️ Added Generators Tests: ${GENERATORS_TESTS}")

add_test(NAME ${GEOMETRY_TESTS} COMMAND ${GEOMETRY_TESTS})
message(STATUS "🛠️ Added Geometry Tests: ${GEOMETRY_TESTS}")

add_test(NAME ${PARTICLE_TESTS} COMMAND ${PARTICLE_TESTS})
message(STATUS "🛠️ Added Particle Tests: ${PARTICLE_TESTS}")

add_test(NAME ${UTILITIES_TESTS} COMMAND ${UTILITIES_TESTS})
message(STATUS "🛠️ Added Utilities Tests: ${UTILITIES_TESTS}")


add_executable(${TARGET_NAME}
    main.cpp

    ${SOURCES}
    ${TEST_SOURCES}
)

# Disable -Werror for GoogleTest
target_compile_options(gtest PRIVATE -Wno-error)
target_compile_options(gmock PRIVATE -Wno-error)
message(STATUS "⚙️  Applying flag '-Wno-error' for the GoogleTest and GoogleMock libraries.")

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

message (STATUS "✅ Build configuration for testing completed successfully. Initiating test suites: [${ALL_TEST_SUITES}]")
