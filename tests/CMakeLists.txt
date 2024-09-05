cmake_minimum_required(VERSION 3.14)
project(Tests)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -O3 -Wall -Wextra -Werror -Wno-maybe-uninitialized")

find_package(Threads REQUIRED)

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/release-1.10.0.tar.gz
)
FetchContent_MakeAvailable(googletest)

if (WIN32)
    find_library(GMSH_LIBRARIES
                NAMES gmsh
                PATHS "C:/Program Files/gmsh-4.12.2-Windows64-sdk/lib"
                NO_DEFAULT_PATH)
    find_package(CGAL REQUIRED)
    if(CGAL_FOUND)
        set(CGAL_NAMESPACE CGAL::CGAL)
        message(STATUS "CGAL library found: ${CGAL_LIBRARIES}")
    endif()
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

if(MSVC)
    find_package(GMP REQUIRED)
    if(GMP_FOUND)
        message(STATUS "GMP found\nGMP libraries: ${GMP_LIBRARIES}")
    endif()
else()
    set(GMP_LIBRARIES gmp)
endif()
include_directories(${GMP_INCLUDE_DIR})

find_package(TBB QUIET)
if (TBB_FOUND)
    set(TBB_NAMESPACE TBB::tbb)
    message(STATUS "TBB library found")
endif()
include_directories(${TBB_INCLUDE_DIRS})

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package(HDF5 REQUIRED COMPONENTS CXX)
include_directories(${HDF5_INCLUDE_DIRS})

if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fp-model strict")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frounding-math")
else()
  message(WARNING "Unsupported compiler. You may need to manually set the appropriate flags for strict floating-point rounding.")
endif()

include_directories("../include/")
add_executable(
    all_tests
    main.cpp
    
    CellSelectorTests.cpp
    ConfigParserTests.cpp
    HDF5HandlerTests.cpp
    MathVectorTests.cpp
    MeshTests.cpp
    ParticleTests.cpp
    RayTriangleIntersectionTests.cpp
    RealNumberGeneratorTests.cpp
    VolumeCreatorTests.cpp
    TetrahedronMeshManagerTests.cpp

    ../src/Utilities/ConfigParser.cpp
    ../src/DataHandling/HDF5Handler.cpp
    ../src/DataHandling/TetrahedronMeshManager.cpp
    ../src/Geometry/MathVector.cpp
    ../src/Geometry/Mesh.cpp
    ../src/Particles/Particle.cpp
    ../src/Geometry/RayTriangleIntersection.cpp
    ../src/Generators/RealNumberGenerator.cpp
    ../src/Utilities/Utilities.cpp
    ../src/Generators/VolumeCreator.cpp
)

if (WIN32)
    set(MPIEXEC_EXECUTABLE "C:/Program Files/Microsoft MPI/Bin/mpiexec.exe")
    set(MPI_CXX_HEADER_DIR "C:/Program Files (x86)/Microsoft SDKs/MPI/Include")
    set(MPI_CXX_LIB_NAMES "msmpi")
    set(MPI_msmpi_LIBRARY "C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64/msmpi.lib")
    set(MPI_CXX_LIBRARIES "C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64/msmpi.lib")
    set(MPI_CXX_COMPILER "cl.exe")
    include_directories(${MPI_CXX_HEADER_DIR})
    link_directories(${MPI_CXX_LIBRARIES})
else()
    find_package(MPI REQUIRED)
    if(MPI_FOUND)
        message(STATUS "MPI libraries: ${MPI_LIBRARIES}")
    endif()
endif()
include_directories(${MPI_INCLUDE_PATH})

if(NOT "$ENV{TRILINOS_DIR}" STREQUAL "")
  	set(TRILINOS_DIR "$ENV{TRILINOS_DIR}")
 	find_package(Trilinos REQUIRED PATHS "${TRILINOS_DIR}")
  	message(STATUS "Trilinos directory set to custom environment variable")
else()
  	find_package(Trilinos REQUIRED PATHS "/usr/local/lib/cmake/Trilinos")
  	message(STATUS "Trilinos directory set to default location")
endif()

if (Trilinos_FOUND)
    message(STATUS "Trilinos libraries: ${Trilinos_LIBRARIES}")
    message(STATUS "Trilinos include dirs: ${Trilinos_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "Trilinos not found")
endif()
include_directories(${Trilinos_INCLUDE_DIRS})
link_directories(${Trilinos_LIBRARY_DIRS})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG)
    message(STATUS "*** Compiling in DEBUG mode ***")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_definitions(RELEASE)
    message(STATUS "*** Compiling in RELEASE mode ***")
else()
    message(WARNING "Unknown build type: ${CMAKE_BUILD_TYPE}")
endif()

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

target_link_libraries(
    all_tests
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

include(CTest)
enable_testing()

add_test(
    NAME all_tests
    COMMAND all_tests
)
