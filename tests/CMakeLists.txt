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
    
    ConfigParserTests.cpp
    HDF5HandlerTests.cpp
    MathVectorTests.cpp
    MeshTests.cpp
    ParticleTests.cpp
    RayTriangleIntersectionTests.cpp
    RealNumberGeneratorTests.cpp
    VolumeCreatorTests.cpp
    VolumetricMeshDataTests.cpp

    ../src/Utilities/ConfigParser.cpp
    ../src/DataHandling/HDF5Handler.cpp
    ../src/DataHandling/VolumetricMeshData.cpp
    ../src/Geometry/MathVector.cpp
    ../src/Geometry/Mesh.cpp
    ../src/Particles/Particle.cpp
    ../src/Geometry/RayTriangleIntersection.cpp
    ../src/Generators/RealNumberGenerator.cpp
    ../src/Utilities/Utilities.cpp
    ../src/Generators/VolumeCreator.cpp
)

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
)

include(CTest)
enable_testing()

# Adding tests
add_test(
    NAME all_tests
    COMMAND all_tests
)
