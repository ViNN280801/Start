cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

###############################################################
#                   Setting compilers                         #
###############################################################

# ══► Helper function to check, set, and print compiler information ◄══
function(set_compiler_if_not_defined env_var cmake_var default_value description)
    if(NOT ${cmake_var})
        if(NOT DEFINED ENV{${env_var}})
            set(ENV{${env_var}} ${default_value})
            message(STATUS "Default ${description} set to ${default_value}")
        else()
            message(STATUS "Using environment variable ${env_var}: $ENV{${env_var}}")
        endif()
        set(${cmake_var} $ENV{${env_var}} PARENT_SCOPE)
    else()
        message(STATUS "CMake ${description} set from command line: ${${cmake_var}}")
    endif()

    # ✦ Print version information ✦
    execute_process(
        COMMAND ${${cmake_var}} --version
        OUTPUT_VARIABLE COMPILER_VERSION
        ERROR_VARIABLE COMPILER_ERROR
        RESULT_VARIABLE COMPILER_RESULT
    )
    if(COMPILER_RESULT EQUAL 0)
        message(STATUS "${description} version: ${COMPILER_VERSION}")

        # Extract the major version from the compiler output (assuming it's in a format like "gcc (Ubuntu 14.0.0) ...")
        string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" VERSION_MATCH ${COMPILER_VERSION})
        string(REGEX MATCH "[0-9]+" MAJOR_VERSION ${VERSION_MATCH})
        set(MAJOR_VERSION ${MAJOR_VERSION} PARENT_SCOPE)
    else()
        message(WARNING "Failed to detect ${description} version: ${COMPILER_ERROR}")
    endif()
endfunction()

# ──► Setting compilers ◄──
set_compiler_if_not_defined(CC CMAKE_C_COMPILER "/usr/bin/gcc-13" "C compiler")
set_compiler_if_not_defined(CXX CMAKE_CXX_COMPILER "/usr/bin/g++-13" "C++ compiler")
set_compiler_if_not_defined(CUDAHOSTCXX CUDAHOSTCXX "/usr/bin/g++-13" "CUDA host C++ compiler")
set_compiler_if_not_defined(CUDA_COMPILER CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" "CUDA compiler")

###############################################################
#              Handling GCC version for CUDA flags            #
###############################################################

# Extract GCC major version
set(gcc_major_version ${MAJOR_VERSION})
set(gxx_major_version ${MAJOR_VERSION})

# ✦ Check if GCC 14 is used, and add the appropriate CUDA flag ✦
if(gcc_major_version STREQUAL "14" OR gxx_major_version STREQUAL "14")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
    message(WARNING "⚠️  GCC 14 or G++ 14 detected. Adding '-allow-unsupported-compiler' flag to CMake CUDA flags. CUDA support for GCC 14 is not guaranteed!")
endif()

###############################################################
#                   End of setting compilers                  #
###############################################################



###############################################################
#                 CUDA Toolkit Directory Setup                #
###############################################################

# ══► Function to check if a directory exists and handle errors ◄══
function(check_directory_exists dir description)
    if(NOT EXISTS ${dir})
        message(FATAL_ERROR "❌ ${description} not found at ${dir}. Please make sure it's installed or correctly configured.")
    else()
        message(STATUS "✅ ${description} located at ${dir}")
    endif()
endfunction()
###############################################################
#           Setting CUDA Toolkit Directory and Compiler        #
###############################################################

# ──► Check if CUDA_TOOLKIT_ROOT_DIR is set by the user ◄──
if(NOT CUDA_TOOLKIT_ROOT_DIR)
    # Set to default if not set
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
    message(STATUS "⚙️  CUDA toolkit root directory not specified, setting default: ${CUDA_TOOLKIT_ROOT_DIR}")
else()
    message(STATUS "⚙️  CUDA toolkit root directory specified: ${CUDA_TOOLKIT_ROOT_DIR}")
endif()

# ✦ Verify that the directory exists ✦
check_directory_exists(${CUDA_TOOLKIT_ROOT_DIR} "CUDA toolkit root directory")

###############################################################
#               Setting CUDA Compiler Path (Optional)         #
###############################################################
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
message(STATUS "⚙️  CUDA compiler set to: ${CMAKE_CUDA_COMPILER}")
###############################################################
#                    End of CUDA Setup                        #
###############################################################
project(Start
        VERSION 1.0
        DESCRIPTION "The Start project is dedicated to the development and testing of \
        a software prototype for optimizing the technological process in vacuum-plasma installations. \
        This software aims to address the current limitations in accurately obtaining process \
        parameters in vacuum-plasma installations with non-rectangular geometries, due to the \
        lack of algorithms for calculating non-standard grids of the electrostatic field generated \
        by plasma particles."
        LANGUAGES CXX CUDA)
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
        message(STATUS "✅ GMSH include directories found at: ${GMSH_INC}")
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
        message(STATUS "✅ GMSH found: Include directories at ${GMSH_INC}, Libraries at ${GMSH_LIBRARIES}")
    endif()

    include_directories(${GMSH_INC})
endif()
###############################################################
#                     CGAL Setup                              #
###############################################################
if(WIN32)
    find_package(CGAL REQUIRED)
    if(CGAL_FOUND)
        set(CGAL_NAMESPACE CGAL::CGAL)
        message(STATUS "✅ CGAL found: Include directories at ${CGAL_INCLUDE_DIRS}, Libraries at ${CGAL_LIBRARIES}")
    else()
        message(FATAL_ERROR "❌ CGAL not found. Please ensure CGAL is installed.")
    endif()
    include_directories(${CGAL_INCLUDE_DIRS})
else()
    find_path(CGAL_INCLUDE_DIR NAMES CGAL/AABB_tree.h
        HINTS /usr/include /usr/local/include
        PATH_SUFFIXES CGAL)

    message(STATUS "✅ CGAL include directory: ${CGAL_INCLUDE_DIR}")

    include_directories(${CGAL_INCLUDE_DIR})
endif()
###############################################################
#                     JSON Setup                              #
###############################################################
if(WIN32)
    find_path(JSON_INC NAMES json.hpp
        HINTS $ENV{JSON_INC}
        PATH_SUFFIXES include)

    if(NOT JSON_INC)
        message(FATAL_ERROR "❌ JSON include directory not found. Please ensure JSON_INC is set.")
    else()
        message(STATUS "✅ JSON include directory found at: ${JSON_INC}")
    endif()

    include_directories(${JSON_INC})
else()
    find_path(JSON_INC NAMES json.hpp
        HINTS /usr/include /usr/local/include /usr/include/nlohmann /usr/include/boost)

    if(NOT JSON_INC)
        message(FATAL_ERROR "❌ JSON include directory not found on Linux/Unix.")
    else()
        message(STATUS "✅ JSON include directory found at: ${JSON_INC}")
    endif()

    include_directories(${JSON_INC})
endif()

###############################################################
#                     MPI Setup                               #
###############################################################
find_package(MPI REQUIRED)
if(MPI_FOUND)
    message(STATUS "✅ MPI found: Include directories at ${MPI_INCLUDE_PATH}, Libraries at ${MPI_LIBRARIES}")
else()
    message(FATAL_ERROR "❌ MPI not found. Please ensure MPI is installed and available.")
endif()
include_directories(${MPI_INCLUDE_PATH})
link_directories(${MPI_LIBRARY_DIRS})

###############################################################
#                     GMP Setup                               #
###############################################################
if(MSVC)
    find_package(GMP REQUIRED)
    if(GMP_FOUND)
        message(STATUS "✅ GMP found: Include directories at ${GMP_INCLUDE_DIR}, Libraries at ${GMP_LIBRARIES}")
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
        message(STATUS "✅ GMP found: Include directories at ${GMP_INCLUDE_DIR}, Libraries at ${GMP_LIBRARY}")
    endif()
endif()
include_directories(${GMP_INCLUDE_DIR})
link_libraries(${GMP_LIBRARY})

###############################################################
#                     Boost Setup                             #
###############################################################
cmake_policy(SET CMP0167 NEW)
find_package(Boost REQUIRED)
if(Boost_FOUND)
    message(STATUS "✅ Boost found: Include directories at ${Boost_INCLUDE_DIRS}, Libraries at ${Boost_LIBRARIES}")
else()
    message(FATAL_ERROR "❌ Boost not found. Please ensure Boost is installed and available.")
endif()
include_directories(${Boost_INCLUDE_DIRS})

###############################################################
#                     HDF5 Setup                              #
###############################################################
find_package(HDF5 REQUIRED COMPONENTS CXX)
if(HDF5_FOUND)
    message(STATUS "✅ HDF5 found: Include directories at ${HDF5_INCLUDE_DIRS}, Libraries at ${HDF5_LIBRARIES}")
else()
    message(FATAL_ERROR "❌ HDF5 not found. Please ensure HDF5 is installed and available.")
endif()
include_directories(${HDF5_INCLUDE_DIRS})
###############################################################
#               C++ Extensions and Trilinos Setup             #
###############################################################

# ══► Ensure C++ extensions are disabled globally to avoid warnings from Kokkos ◄══
set(CMAKE_CXX_EXTENSIONS OFF)

###############################################################
#                     Trilinos Setup                          #
###############################################################
if(NOT "$ENV{TRILINOS_DIR}" STREQUAL "")
    set(TRILINOS_DIR "$ENV{TRILINOS_DIR}")
    find_package(Trilinos REQUIRED PATHS "${TRILINOS_DIR}")
    message(STATUS "✅ Trilinos directory set to custom environment variable: ${TRILINOS_DIR}")
else()
    find_package(Trilinos REQUIRED PATHS "/usr/local/lib/cmake/Trilinos")
    message(STATUS "✅ Trilinos directory set to default location")
endif()

if(Trilinos_FOUND)
    message(STATUS "✅ Trilinos libraries: ${Trilinos_LIBRARIES}")
    message(STATUS "✅ Trilinos include directories: ${Trilinos_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "❌ Trilinos not found. Please ensure Trilinos is installed.")
endif()
include_directories(${Trilinos_INCLUDE_DIRS})
link_directories(${Trilinos_LIBRARY_DIRS})

###############################################################
#                     TBB Setup                               #
###############################################################
find_package(TBB)
if(TBB_FOUND)
    set(TBB_NAMESPACE TBB::tbb)
    message(STATUS "✅ TBB found: Include directories at ${TBB_INCLUDE_DIRS}, Libraries at ${TBB_LIBRARIES}")
else()
    message(WARNING "⚠️ TBB not found. Continuing without TBB support.")
endif()
include_directories(${TBB_INCLUDE_DIRS})
###############################################################
#               End of Package Configuration                  #
###############################################################

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║           COMPILER AND BUILD TYPE CONFIGURATION                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# ╔═► MSVC Specific Compile Options ◄═════════════════════════════════════════╗
# ║ If MSVC is used, set compiler options for warning level and optimizations.║
# ║ Remove unnecessary flags in debug mode and optimize in release mode.      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
if(MSVC)
    add_compile_options(/W4)  # Enable warning level 4
    string(REPLACE "/RTC1" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")  # Remove /RTC1 flag
    string(REPLACE "/O2" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")    # Remove /O2 flag from debug
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")                # Optimize in release mode
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")                            # Compiling as big object file
else()
    set(COMMON_CPP_FLAGS "-Wall -Wpedantic -Wextra -pthread -O3 -flarge-source-files -Wmisleading-indentation")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_CPP_FLAGS}")
    message(STATUS "⚙️ Non-MSVC: Added common flags ${COMMON_CPP_FLAGS}")
endif()

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║           FLOATING-POINT ROUNDING MODE CONFIGURATION                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# ╔═► Configure compiler-specific floating-point rounding ◄══════════════════╗
# ║ Detect the compiler being used (Intel, GNU, MSVC) and apply appropriate  ║
# ║ floating-point rounding flags for strict computation control.            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fp-model strict")
    message(STATUS "⚙️ Intel: Using strict floating-point rounding with -fp-model strict")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frounding-math")
    message(STATUS "⚙️ GNU: Using strict floating-point rounding with -frounding-math")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:strict")
    message(STATUS "⚙️ MSVC: Using strict floating-point rounding with /fp:strict")
else()
    message(WARNING "⚠️ Unsupported compiler. You may need to manually set appropriate flags for strict floating-point rounding.")
endif()

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║           BUILD TYPE CONFIGURATION                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# ╔═► Check if we're building in Debug or Release mode ◄═════════════════════╗
# ║ Define the appropriate macros (`DEBUG` or `RELEASE`) based on the build  ║
# ║ type (Debug, Release, etc.).                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG)
    message(STATUS "Compiling in DEBUG mode")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_definitions(RELEASE)
    message(STATUS "Compiling in RELEASE mode")
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
option(START_DEBUG "Compile the project in Start Debug mode" OFF)
option(START_RELEASE "Compile the project in Start Release mode" OFF)

if(START_DEBUG)
    add_compile_definitions(START_DEBUG)
    message(STATUS " Compiling in START_DEBUG mode ")
elseif(START_RELEASE)
    add_compile_definitions(START_RELEASE)
    message(STATUS " Compiling in START_RELEASE mode ")
else()
    message(WARNING "⚠️ Neither START_DEBUG nor START_RELEASE is defined.")
endif()
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║              END OF CONFIGURING OPTIONS                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/DataHandling
    ${CMAKE_SOURCE_DIR}/include/FiniteElementMethod
    ${CMAKE_SOURCE_DIR}/include/FiniteElementMethod/BoundaryConditions
    ${CMAKE_SOURCE_DIR}/include/FiniteElementMethod/Cell
    ${CMAKE_SOURCE_DIR}/include/FiniteElementMethod/Cubature
    ${CMAKE_SOURCE_DIR}/include/Generators
    ${CMAKE_SOURCE_DIR}/include/Geometry
    ${CMAKE_SOURCE_DIR}/include/Particles
    ${CMAKE_SOURCE_DIR}/include/Utilities
)

add_subdirectory(src)

set(CPP_SOURCES
    src/main.cpp
    src/ModelingMainDriver.cpp
)
set(CUDA_SOURCES
    # !!! Not to forget place .cu files here !!!
)
set(SOURCES ${CPP_SOURCES} ${CUDA_SOURCES})

set(EXECUTABLE_NAME nia_start_core)
add_executable(${EXECUTABLE_NAME} ${SOURCES})

set_target_properties(${EXECUTABLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")
target_link_libraries(${EXECUTABLE_NAME} PRIVATE 
    ${CGAL_NAMESPACE}
    ${GMSH_LIBRARIES} 
    ${Boost_LIBRARIES} 
    ${GMP_LIBRARIES}
    ${HDF5_LIBRARIES}
    ${TBB_NAMESPACE}
    ${TBB_LIBRARIES}
    ${MPI_LIBRARIES}
    ${Trilinos_LIBRARIES}
    ${CUDA_LIBRARIES}
    DataHandling
    FiniteElementMethod
    Generators
    Geometry
    Particles
    Utilities
)
