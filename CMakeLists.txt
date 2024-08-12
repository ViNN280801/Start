cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(Start
        VERSION 1.0
        DESCRIPTION "The Start project is dedicated to the development and testing of \
        a software prototype for optimizing the technological process in vacuum-plasma installations. \
        This software aims to address the current limitations in accurately obtaining process \
        parameters in vacuum-plasma installations with non-rectangular geometries, due to the \
        lack of algorithms for calculating non-standard grids of the electrostatic field generated \
        by plasma particles."
        LANGUAGES CXX)

if (WIN32)
    set(GMSH_INC $ENV{GMSH_INC})
    set(CGAL_INC $ENV{CGAL_INC})
    set(JSON_INC $ENV{JSON_INC})
    set(MPI_INC $ENV{MPI_INC})

    message(STATUS "GMSH include directories: ${GMSH_INC}")
    message(STATUS "CGAL include directories: ${CGAL_INC}")
    message(STATUS "Json include directories: ${JSON_INC}")
    message(STATUS "MPI include directories: ${MPI_INC}")

    include_directories(${GMSH_INC} ${CGAL_INC} ${JSON_INC} ${MPI_INC})

    find_library(GMSH_LIBRARIES
                NAMES gmsh
                PATHS "C:/Program Files/gmsh-4.12.2-Windows64-sdk/lib"
                NO_DEFAULT_PATH)
    find_package(CGAL REQUIRED)
    if(CGAL_FOUND)
        set(CGAL_NAMESPACE CGAL::CGAL)
        message(STATUS "CGAL libraries: ${CGAL_LIBRARIES}")
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
    set(GMP_LIBRARIES "C:/Program Files/CGAL-5.6.1/auxiliary/gmp/lib/libgmp-10.lib;C:/Program Files/CGAL-5.6.1/auxiliary/gmp/lib/libmpfr-4.lib")
    message(STATUS "GMP libraries: ${GMP_LIBRARIES}")
    message(STATUS "GMP include directories: ${GMP_INCLUDE_DIR}")
else()
    set(GMP_LIBRARIES gmp)
endif()
include_directories(${GMP_INCLUDE_DIR})

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package(HDF5 REQUIRED COMPONENTS CXX)
include_directories(${HDF5_INCLUDE_DIRS})

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

find_package(TBB)
if (TBB_FOUND)
    set(TBB_NAMESPACE TBB::tbb)
    message(STATUS "TBB libraries: ${TBB_LIBRARIES}")
endif()
include_directories(${TBB_INCLUDE_DIRS})

if (MSVC)
    add_compile_options(/W4)
    string(REPLACE "/RTC1" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    string(REPLACE "/O2" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wpedantic -Wextra -pthread -O3")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fp-model strict")
    message(STATUS "Intel: Rounding floating-point numbers with flag -fp-model strict")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frounding-math")
    message(STATUS "GNU: Rounding floating-point numbers with flag -frounding-math")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:strict")
    message(STATUS "MSVC: Rounding floating-point numbers with flag /fp:strict")
else()
    message(WARNING "Unsupported compiler. You may need to manually set the appropriate flags for strict floating-point rounding.")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_subdirectory(src)

set(SOURCES
    src/main.cpp
    src/ModelingMainDriver.cpp
)

set(EXECUTABLE_NAME nia_start_core)
add_executable(${EXECUTABLE_NAME} ${SOURCES})

set_target_properties(${EXECUTABLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")
target_link_libraries(${EXECUTABLE_NAME} PRIVATE 
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
    DataHandling
    FiniteElementMethod
    Generators
    Geometry
    ParticleTracker
    Particles
    Utilities
)
