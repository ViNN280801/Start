# tests/CMakeLists.txt

option(VECTORIZATION_ON "Enable vectorization and anti-aliasing flags" ON)
option(START_OPENMP_ON "Enable OpenMP support for the project" ON)
option(START_MPI_ON "Enable MPI support for the project" ON)
option(SHOW_LOGS_ON "Enable showing all the logs" ON)
option(GPU_CUDA_ON "Enable CUDA support for GPU acceleration" ON)
option(START_RELEASE "Compile the project in Start Release mode" ON)

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
    endif()
endfunction()

if (GPU_CUDA_ON)
    message(STATUS "✅ CUDA support for GPU acceleration is enabled.")
    add_compile_options(-DUSE_CUDA)

    set(CUDAHOSTCXX "/usr/bin/g++-13")
    message(STATUS "⚙️  Using ${CUDAHOSTCXX} as the CUDA host compiler.")
    set(CMAKE_CUDA_HOST_COMPILER ${CUDAHOSTCXX})

    set(ENV{NVCC_WRAPPER_DEFAULT_COMPILER} ${CUDAHOSTCXX})
    message(STATUS "⚙️  Setting NVCC_WRAPPER_DEFAULT_COMPILER to ${CUDAHOSTCXX}")

    set_compiler_if_not_defined(CMAKE_CUDA_COMPILER CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" "CUDA compiler")

    set(NVCC_WRAPPER_PATH "/usr/local/bin/nvcc_wrapper")
    if (EXISTS ${NVCC_WRAPPER_PATH})
        message(STATUS "⚙️  Using nvcc_wrapper at ${NVCC_WRAPPER_PATH} as the C++ compiler for CUDA-enabled targets.")
        set(CMAKE_CXX_COMPILER ${NVCC_WRAPPER_PATH})

        set(SUPPRESS_FLAGS_FOR_TRILINOS_CUDA "--expt-extended-lambda -Wno-unused-parameter -Wno-dangling-reference -Wno-implicit-fallthrough")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SUPPRESS_FLAGS_FOR_TRILINOS_CUDA}")
        message(STATUS "⚙️  Added flags [${SUPPRESS_FLAGS_FOR_TRILINOS_CUDA}] to the C++ compile flags to suppress some warnings from Trilinos")
    else()
        message(FATAL_ERROR "❌ nvcc_wrapper not found at ${NVCC_WRAPPER_PATH}. Please ensure it is installed.")
    endif()

    ###############################################################
    #                 CUDA Toolkit Directory Setup                #
    ###############################################################
    # ══► Function to check if a directory exists and handle errors ◄══
    function(check_directory_exists dir description)
        if(NOT EXISTS ${dir})
            message(FATAL_ERROR "❌ ${description} not found at [${dir}]. Please make sure it's installed or correctly configured.")
        else()
            message(STATUS "✅ ${description} located at [${dir}]")
        endif()
    endfunction()
    ###############################################################
    #           Setting CUDA Toolkit Directory and Compiler        #
    ###############################################################

    # ──► Check if CUDA_TOOLKIT_ROOT_DIR is set by the user ◄──
    if(NOT CUDA_TOOLKIT_ROOT_DIR)
        # Set to default if not set
        set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
        set(CUDA_INCLUDE_DIR "${CUDA_TOOLKIT_ROOT_DIR}/include")
        
        message(STATUS "⚙️  CUDA toolkit root directory not specified, setting default: [${CUDA_TOOLKIT_ROOT_DIR}]. Include directories: [${CUDA_INCLUDE_DIR}]")
    else()
        message(STATUS "⚙️  CUDA toolkit root directory specified: [${CUDA_TOOLKIT_ROOT_DIR}]")
    endif()

    # ✦ Verify that the directory exists ✦
    check_directory_exists(${CUDA_TOOLKIT_ROOT_DIR} "CUDA toolkit root directory")

    ###############################################################
    #               Setting CUDA Compiler Path (Optional)         #
    ###############################################################

    # Include CUDA as a language in the project only if GPU_CUDA_ON is enabled
    set(CUDA_ARCHITECTURES "52 61 75 86 89")
    set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})
    message(STATUS "⚙️  CUDA support enabled with CUDA_ARCHITECTURES={${CUDA_ARCHITECTURES}}")
    
    set(SUPPRESS_DEPRECATED_DECLARATIONS_WARNING_FLAG "-Wno-deprecated-declarations")
    message(STATUS "⚙️  Setting flag '${SUPPRESS_DEPRECATED_DECLARATIONS_WARNING_FLAG}' to suppress warning: ‘template<class _Tp> struct std::is_pod’ is deprecated: use 'is_standard_layout && is_trivial'")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SUPPRESS_DEPRECATED_DECLARATIONS_WARNING_FLAG} -Wno-maybe-uninitialized -Wno-parentheses")

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -Xcompiler -mno-amx-bf16 -Xcompiler -mno-amx-int8 -Xcompiler -mno-amx-tile")
    message(STATUS "⚙️  Compiling with CUDA flags: [${CMAKE_CUDA_FLAGS}]")
    ###############################################################
    #                    End of CUDA Setup                        #
    ###############################################################

    enable_language(CUDA)

    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    message(STATUS "⚙️  Using C++${CMAKE_CUDA_STANDARD} standard for the CUDA.")
    include_directories(${CUDA_INCLUDE_DIR})
else()
    message(STATUS "🚫 CUDA support for GPU acceleration is disabled.")
endif()

if(VECTORIZATION_ON)
    message(STATUS "✅ Vectorization and anti-aliasing flags are enabled.")
else()
    message(STATUS "🚫 Vectorization and anti-aliasing flags are disabled.")
endif()

if(START_OPENMP_ON)
    message(STATUS "✅ OpenMP support enabled for the project.")
    add_compile_options(-fopenmp -DUSE_OMP)
else()
    message(STATUS "🚫 OpenMP support disabled for the project.")
endif()

if (START_MPI_ON)
    message(STATUS "✅ MPI is enabled for the project.")
    set(DISABLE_CAST_BETWEEN_INCOMPATIBLE_FUNC_TYPES_FLAG "-Wno-cast-function-type")
    add_compile_options(${DISABLE_CAST_BETWEEN_INCOMPATIBLE_FUNC_TYPES_FLAG} -DUSE_MPI)
else()
    message(STATUS "🚫 MPI is disabled for the project.")
endif()

# Initialize variables for optional flags
set(ANTI_ALIASING_FLAG "")
set(VECTORIZATION_FLAGS "")

# Checking support of the AVX, AVX2 and AVX-512
execute_process(
    COMMAND lscpu
    OUTPUT_VARIABLE CPU_INFO
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if (CPU_INFO MATCHES "avx512")
    set(VECTOR_WIDTH 512)
elseif (CPU_INFO MATCHES "avx" OR CPU_INFO MATCHES "avx2")
    set(VECTOR_WIDTH 256)
else()
    set(VECTOR_WIDTH 128)
endif()

# Check if vectorization is enabled
if(VECTORIZATION_ON)
    # Check for Clang and set specific flags
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(ANTI_ALIASING_FLAG "-fstrict-aliasing")
        set(VECTORIZATION_FLAGS "-fvectorize")
        set(CLANG_FLAGS "-Xclang" "-flarge-source-files")
        add_compile_options(${CLANG_FLAGS} ${VECTORIZATION_FLAGS} ${ANTI_ALIASING_FLAG})
        add_compile_options("-mprefer-vector-width=${VECTOR_WIDTH}")
        message(STATUS "⚙️  Clang-specific vectorization flags added: [${CLANG_FLAGS} ${VECTORIZATION_FLAGS} ${ANTI_ALIASING_FLAG}]")

    # Check for GCC and set specific flags
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(ANTI_ALIASING_FLAG "-fstrict-aliasing")
        set(VECTORIZATION_FLAGS "-ftree-vectorize")
        set(GCC_FLAGS "-flarge-source-files")
        add_compile_options(${GCC_FLAGS} ${VECTORIZATION_FLAGS} ${ANTI_ALIASING_FLAG})
        add_compile_options("-mprefer-vector-width=${VECTOR_WIDTH}")
        message(STATUS "⚙️  GCC-specific vectorization flags added: [${GCC_FLAGS} ${VECTORIZATION_FLAGS} ${ANTI_ALIASING_FLAG}]")

    # Check for Intel compiler and set specific flags
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        set(ANTI_ALIASING_FLAG "-ansi-alias")
        set(VECTORIZATION_FLAGS "-restrict" "-xHost" "-vecabi=cmdtarget" "-qopt-zmm-usage=high")
        set(INTEL_COMPILER_FLAGS "-flarge-source-files")
        add_compile_options(${INTEL_COMPILER_FLAGS} ${VECTORIZATION_FLAGS} ${ANTI_ALIASING_FLAG})
        message(STATUS "⚙️  Intel-specific vectorization flags added: [${INTEL_COMPILER_FLAGS} ${VECTORIZATION_FLAGS} ${ANTI_ALIASING_FLAG}]")
    endif()
    message(STATUS "⚙️  Vector width set to: ${VECTOR_WIDTH}")
endif()

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

# ##############################################################################
# CGAL Setup                              #
# ##############################################################################
find_package(CGAL REQUIRED)

if(CGAL_FOUND)
  message(STATUS "✅ CGAL found: ${CGAL_INCLUDE_DIRS}")
  include_directories(${CGAL_INCLUDE_DIRS})
else()
  message(FATAL_ERROR "❌ CGAL not found.")
endif()

include_directories(${CGAL_INCLUDE_DIR})

# ##############################################################################
# GMP Setup                               #
# ##############################################################################
find_path(GMP_INCLUDE_DIR gmp.h HINTS /usr/include/x86_64-linux-gnu
                                      /usr/include/CGAL)

find_library(
  GMP_LIBRARY
  NAMES gmp libgmp
  HINTS /usr/lib/x86_64-linux-gnu)

if(NOT GMP_INCLUDE_DIR OR NOT GMP_LIBRARY)
  message(FATAL_ERROR "❌ GMP not found. Please ensure GMP is installed.")
else()
  message(
    STATUS
      "✅ GMP found: Include directories at [${GMP_INCLUDE_DIR}], Libraries: [${GMP_LIBRARY}]"
  )
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
    message(STATUS "⚙️  Compiling in DEBUG mode")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -O3 -Wall -Wextra -Wno-maybe-uninitialized -Wmisleading-indentation -flarge-source-files")

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
    ${DATA_HANDLING_SRC_PATH}/TriangleMeshHdf5Manager.cpp
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
    ${FINITE_ELEMENT_METHOD_SRC_PATH}/GSMAssembler.cpp
    ${FINITE_ELEMENT_METHOD_SRC_PATH}/MatrixEquationSolver.cpp
)
set(GENERATORS_SOURCES
    ${GENERATORS_SRC_PATH}/Host/ParticleGeneratorHost.cpp
    ${GENERATORS_SRC_PATH}/Host/RealNumberGeneratorHost.cpp
)
set(GEOMETRY_SOURCES
    ${GEOMETRY_SRC_PATH}/Mesh.cpp
    ${GEOMETRY_SRC_PATH}/RayTriangleIntersection.cpp
)
set(PARTICLES_SOURCES
    ${PARTICLES_SRC_PATH}/Particle.cpp

    ${PARTICLES_SRC_PATH}/PhysicsCore/CollisionModel/CollisionModelFactory.cpp
    ${PARTICLES_SRC_PATH}/PhysicsCore/CollisionModel/HSModel.cpp
    ${PARTICLES_SRC_PATH}/PhysicsCore/CollisionModel/VHSModel.cpp
    ${PARTICLES_SRC_PATH}/PhysicsCore/CollisionModel/VSSModel.cpp
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
    DataHandling/TriangleMeshHdf5ManagerTests.cpp
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
    Generators/ParticleGeneratorTests.cpp
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

if (START_OPENMP_ON)
    target_link_libraries(${TARGET_NAME} PRIVATE ${OpenMP_LIBRARIES})
endif()
if (GPU_CUDA_ON)
    target_compile_options(${TARGET_NAME} PRIVATE "-mno-amx-bf16" "-mno-amx-int8" "-mno-amx-tile")
    target_link_libraries(${TARGET_NAME} PRIVATE ${CUDA_LIBRARIES})
    set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()


include(CTest)
enable_testing()

message (STATUS "✅ Build configuration for testing completed successfully. Initiating test suites: [${ALL_TEST_SUITES}]")
