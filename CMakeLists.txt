# CMakeLists.txt
cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# --------{---(@    Setting custom options with default values   @}}>-----
option(GPU_CUDA_ON "Enable CUDA support for GPU acceleration" ON)
option(VECTORIZATION_ON "Enable vectorization and anti-aliasing flags" ON)
option(START_OPENMP_ON "Enable OpenMP support for the project" ON)
option(START_MPI_ON "Enable MPI support for the project" ON)
option(SHOW_LOGS_ON "Enable showing all the logs" ON)
option(START_DEBUG "Compile the project in Start Debug mode" OFF)
option(START_RELEASE "Compile the project in Start Release mode" ON)
# --------{---(@    Setting custom options with default values   @}}>-----

# ##############################################################################
# °º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸
# ##############################################################################
# #########                    Reporting about custom project flags ##########
if(BUILD_SHARED_LIBS)
  add_compile_options(-fPIC)
endif()

if(GPU_CUDA_ON)
  message(STATUS "✅ CUDA support for GPU acceleration is enabled.")
  add_compile_options(-DUSE_CUDA)
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
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -DUSE_OMP")
else()
  message(STATUS "🚫 OpenMP support disabled for the project.")
endif()

if(START_MPI_ON)
  message(STATUS "✅ MPI is enabled for the project.")
  set(DISABLE_CAST_BETWEEN_INCOMPATIBLE_FUNC_TYPES_FLAG
      "-Wno-cast-function-type")
  add_compile_options(${DISABLE_CAST_BETWEEN_INCOMPATIBLE_FUNC_TYPES_FLAG}
                      -DUSE_MPI)
else()
  message(STATUS "🚫 MPI is disabled for the project.")
endif()

if(SHOW_LOGS_ON)
  message(STATUS "✅ Logging is enabled.")
  add_compile_options(-DSHOW_LOGS)
else()
  message(STATUS "🚫 Logging is disabled.")
endif()
# ##############################################################################
# °º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸
# ##############################################################################

# ##############################################################################
# Setting compilers                         #
# ##############################################################################

# ══► Helper function to check, set, and print compiler information ◄══
function(set_compiler_if_not_defined env_var cmake_var default_value
         description)
  if(NOT ${cmake_var})
    if(NOT DEFINED ENV{${env_var}})
      set(ENV{${env_var}} ${default_value})
      message(STATUS "Default ${description} set to ${default_value}")
    else()
      message(STATUS "Using environment variable ${env_var}: $ENV{${env_var}}")
    endif()
    set(${cmake_var}
        $ENV{${env_var}}
        PARENT_SCOPE)
  else()
    message(
      STATUS "CMake ${description} set from command line: ${${cmake_var}}")
  endif()

  # ✦ Print version information ✦
  execute_process(
    COMMAND ${${cmake_var}} --version
    OUTPUT_VARIABLE COMPILER_VERSION
    ERROR_VARIABLE COMPILER_ERROR
    RESULT_VARIABLE COMPILER_RESULT)
  if(COMPILER_RESULT EQUAL 0)
    message(STATUS "${description} version: ${COMPILER_VERSION}")

    # Extract the major version from the compiler output (assuming it's in a
    # format like "gcc (Ubuntu 14.0.0) ...")
    string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" VERSION_MATCH
                 ${COMPILER_VERSION})
    string(REGEX MATCH "[0-9]+" MAJOR_VERSION ${VERSION_MATCH})
    set(MAJOR_VERSION
        ${MAJOR_VERSION}
        PARENT_SCOPE)
  endif()
endfunction()

# ──► Setting compilers ◄──
set_compiler_if_not_defined(CC CMAKE_C_COMPILER "/usr/bin/gcc" "C compiler")
set_compiler_if_not_defined(CXX CMAKE_CXX_COMPILER "/usr/bin/g++"
                            "C++ compiler")

if(GPU_CUDA_ON)
  set(CUDAHOSTCXX "/usr/bin/g++")
  message(STATUS "⚙️  Using ${CUDAHOSTCXX} as the CUDA host compiler.")
  set(CMAKE_CUDA_HOST_COMPILER ${CUDAHOSTCXX})

  set(ENV{NVCC_WRAPPER_DEFAULT_COMPILER} ${CUDAHOSTCXX})
  message(STATUS "⚙️  Setting NVCC_WRAPPER_DEFAULT_COMPILER to ${CUDAHOSTCXX}")

  set_compiler_if_not_defined(CMAKE_CUDA_COMPILER CMAKE_CUDA_COMPILER
                              "/usr/local/cuda/bin/nvcc" "CUDA compiler")

  set(NVCC_WRAPPER_PATH "/usr/local/bin/nvcc_wrapper")
  if(EXISTS ${NVCC_WRAPPER_PATH})
    message(
      STATUS
        "⚙️  Using nvcc_wrapper at ${NVCC_WRAPPER_PATH} as the C++ compiler for CUDA-enabled targets."
    )
    set(CMAKE_CXX_COMPILER ${NVCC_WRAPPER_PATH})

    set(SUPPRESS_FLAGS_FOR_TRILINOS_CUDA
        "--expt-extended-lambda -Wno-unused-parameter -Wno-dangling-reference -Wno-implicit-fallthrough"
    )
    set(CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} ${SUPPRESS_FLAGS_FOR_TRILINOS_CUDA}")
    message(
      STATUS
        "⚙️  Added flags [${SUPPRESS_FLAGS_FOR_TRILINOS_CUDA}] to the C++ compile flags to suppress some warnings from Trilinos"
    )
  else()
    message(
      FATAL_ERROR
        "❌ nvcc_wrapper not found at ${NVCC_WRAPPER_PATH}. Please ensure it is installed."
    )
  endif()
endif()

# ##############################################################################
# Handling GCC version for CUDA flags            #
# ##############################################################################

# Extract GCC major version
set(gcc_major_version ${MAJOR_VERSION})
set(gxx_major_version ${MAJOR_VERSION})

# ✦ Check if GCC 14 is used, and add the appropriate CUDA flag ✦
if(gcc_major_version STREQUAL "14" OR gxx_major_version STREQUAL "14")
  message(STATUS "⚙️  Using GCC-14")
  if(GPU_CUDA_ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
    message(
      WARNING
        "⚠️  Using GCC ${gcc_major_version} with CUDA. Adding '-allow-unsupported-compiler' flag to CMake CUDA flags. CUDA support for GCC ${gcc_major_version} is not guaranteed!"
    )
  endif()
endif()

# ##############################################################################
# End of setting compilers                  #
# ##############################################################################

if(GPU_CUDA_ON)
  # ############################################################################
  # CUDA Toolkit Directory Setup                #
  # ############################################################################
  # ══► Function to check if a directory exists and handle errors ◄══
  function(check_directory_exists dir description)
    if(NOT EXISTS ${dir})
      message(
        FATAL_ERROR
          "❌ ${description} not found at [${dir}]. Please make sure it's installed or correctly configured."
      )
    else()
      message(STATUS "✅ ${description} located at [${dir}]")
    endif()
  endfunction()
  # ############################################################################
  # Setting CUDA Toolkit Directory and Compiler        #
  # ############################################################################

  # ──► Check if CUDA_TOOLKIT_ROOT_DIR is set by the user ◄──
  if(NOT CUDA_TOOLKIT_ROOT_DIR)
    # Set to default if not set
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
    set(CUDA_INCLUDE_DIR "${CUDA_TOOLKIT_ROOT_DIR}/include")

    message(
      STATUS
        "⚙️  CUDA toolkit root directory not specified, setting default: [${CUDA_TOOLKIT_ROOT_DIR}]. Include directories: [${CUDA_INCLUDE_DIR}]"
    )
  else()
    message(
      STATUS
        "⚙️  CUDA toolkit root directory specified: [${CUDA_TOOLKIT_ROOT_DIR}]")
  endif()

  # ✦ Verify that the directory exists ✦
  check_directory_exists(${CUDA_TOOLKIT_ROOT_DIR} "CUDA toolkit root directory")

  # ############################################################################
  # Setting CUDA Compiler Path (Optional)         #
  # ############################################################################

  # Include CUDA as a language in the project only if GPU_CUDA_ON is enabled
  set(CMAKE_CUDA_ARCHITECTURES "89")
  set_property(GLOBAL PROPERTY CMAKE_CUDA_ARCHITECTURES
                               ${CMAKE_CUDA_ARCHITECTURES})
  message(
    STATUS
      "⚙️  CUDA support enabled with CMAKE_CUDA_ARCHITECTURES={${CMAKE_CUDA_ARCHITECTURES}}"
  )

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
  message(STATUS "⚙️  Compiling with CUDA flags: [${CMAKE_CUDA_FLAGS}]")
  # ############################################################################
  # End of CUDA Setup                        #
  # ############################################################################
endif()

project(
  Start
  VERSION 1.0
  DESCRIPTION
    "The Start project is dedicated to the development and testing of \
        a software prototype for optimizing the technological process in vacuum-plasma installations. \
        This software aims to address the current limitations in accurately obtaining process \
        parameters in vacuum-plasma installations with non-rectangular geometries, due to the \
        lack of algorithms for calculating non-standard grids of the electrostatic field generated \
        by plasma particles."
  LANGUAGES CXX)
if(GPU_CUDA_ON)
  enable_language(CUDA)
endif()

# Global C++ standard setting
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Global CUDA include directories if GPU support is enabled
if(GPU_CUDA_ON)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  message(STATUS "⚙️  Using C++${CMAKE_CUDA_STANDARD} standard for the CUDA.")
  include_directories(${CUDA_INCLUDE_DIR})
endif()

message(STATUS "⚙️  Using C++${CMAKE_CXX_STANDARD} standard.")

# ##############################################################################
# GMSH Setup                              #
# ##############################################################################
find_path(
  GMSH_INC
  NAMES gmsh.h
  HINTS /usr/include /usr/local/include)

find_library(
  GMSH_LIBRARIES
  NAMES gmsh
  HINTS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)

if(NOT GMSH_INC OR NOT GMSH_LIBRARIES)
  message(FATAL_ERROR "❌ GMSH not found on Linux/Unix.")
else()
  message(
    STATUS
      "✅ GMSH found: Include directories at [${GMSH_INC}], Libraries at [${GMSH_LIBRARIES}]"
  )
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
# OpenMP Setup                            #
# ##############################################################################
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
  message(
    STATUS
      "✅ OpenMP found: Include directories at [${OpenMP_INCLUDE_PATH}], Libraries at [${OpenMP_LIBRARY_DIRS}]. Libraries: [${OpenMP_LIBRARIES}]"
  )
else()
  message(
    FATAL_ERROR
      "❌ OpenMP not found. Please ensure OpenMP is installed and available.")
endif()
include_directories(${OpenMP_INCLUDE_PATH})
link_directories(${OpenMP_LIBRARY_DIRS})

# ##############################################################################
# JSON Setup                              #
# ##############################################################################
find_path(
  JSON_INC
  NAMES json.hpp
  HINTS /usr/include /usr/local/include /usr/include/nlohmann
        /usr/include/boost)

if(NOT JSON_INC)
  message(FATAL_ERROR "❌ JSON include directory not found on Linux/Unix.")
else()
  message(STATUS "✅ JSON include directory found at: [${JSON_INC}]")

endif()

include_directories(${JSON_INC})
# ##############################################################################
# MPI Setup                               #
# ##############################################################################
find_package(MPI REQUIRED)
if(MPI_FOUND)
  message(
    STATUS
      "✅ MPI found: Include directories at [${MPI_INCLUDE_PATH}], Libraries: [${MPI_LIBRARIES}]"
  )
else()
  message(
    FATAL_ERROR "❌ MPI not found. Please ensure MPI is installed and available."
  )
endif()
include_directories(${MPI_INCLUDE_PATH})
link_directories(${MPI_LIBRARY_DIRS})

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

# ##############################################################################
# Boost Setup                             #
# ##############################################################################
find_package(Boost REQUIRED)
if(Boost_FOUND)
  message(
    STATUS
      "✅ Boost found: Include directories at [${Boost_INCLUDE_DIRS}], Libraries: [${Boost_LIBRARIES}]"
  )
else()
  message(
    FATAL_ERROR
      "❌ Boost not found. Please ensure Boost is installed and available.")
endif()
include_directories(${Boost_INCLUDE_DIRS})

# ##############################################################################
# HDF5 Setup                              #
# ##############################################################################
find_package(HDF5 REQUIRED COMPONENTS CXX)
if(HDF5_FOUND)
  message(
    STATUS
      "✅ HDF5 found: Include directories at [${HDF5_INCLUDE_DIRS}], Libraries: [${HDF5_LIBRARIES}]"
  )
else()
  message(
    FATAL_ERROR
      "❌ HDF5 not found. Please ensure HDF5 is installed and available.")
endif()
include_directories(${HDF5_INCLUDE_DIRS})
# ##############################################################################
# C++ Extensions and Trilinos Setup             #
# ##############################################################################

# ══► Ensure C++ extensions are disabled globally to avoid warnings from Kokkos
# ◄══
set(CMAKE_CXX_EXTENSIONS OFF)
message(STATUS "⚙️  C++ extensions set to OFF, because of Kokkos requirements.")

# ##############################################################################
# Trilinos Setup                          #
# ##############################################################################
if(NOT "$ENV{TRILINOS_DIR}" STREQUAL "")
  set(TRILINOS_DIR "$ENV{TRILINOS_DIR}")
  find_package(
    Trilinos
    REQUIRED
    Belos
    Intrepid2
    Kokkos
    KokkosKernels
    MueLu
    Shards
    Teuchos
    Tpetra
    PATHS
    "${TRILINOS_DIR}")
  message(
    STATUS
      "✅ Trilinos directory set to custom environment variable: [${TRILINOS_DIR}]"
  )
else()
  set(TRILINOS_DIR "/usr/local/lib/cmake/Trilinos")
  find_package(
    Trilinos
    REQUIRED
    Belos
    Intrepid2
    Kokkos
    KokkosKernels
    MueLu
    Shards
    Teuchos
    Tpetra
    PATHS
    "${TRILINOS_DIR}")
  message(
    STATUS "✅ Trilinos directory set to default location: [${TRILINOS_DIR}]")
endif()

if(Trilinos_FOUND)
  message(STATUS "✅ Trilinos libraries: [${Trilinos_LIBRARIES}]")
  message(STATUS "✅ Trilinos include directories: [${Trilinos_INCLUDE_DIRS}]")
else()
  message(
    FATAL_ERROR "❌ Trilinos not found. Please ensure Trilinos is installed.")
endif()
include_directories(${Trilinos_INCLUDE_DIRS})
link_directories(${Trilinos_LIBRARY_DIRS})

# ##############################################################################
# TBB Setup                               #
# ##############################################################################
find_package(TBB)
if(TBB_FOUND)
  set(TBB_NAMESPACE TBB::tbb)
  message(
    STATUS
      "✅ TBB found: Include directories at [${TBB_INCLUDE_DIRS}], Libraries: [${TBB_LIBRARIES}]"
  )
else()
  message(WARNING "⚠️ TBB not found. Continuing without TBB support.")
endif()
include_directories(${TBB_INCLUDE_DIRS})
# ##############################################################################
# End of Package Configuration                  #
# ##############################################################################

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║           COMPILER AND BUILD TYPE CONFIGURATION                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# ╔═► Specific Compile Options ◄══════════════════════════════════════════════╗
# ║ Set compiler options for warning level and optimizations.                 ║
# ║ Remove unnecessary flags in debug mode and optimize in release mode.      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
set(SUPRESS_SOME_WARNINGS
    "-Wno-deprecated-declarations -Wno-unused-parameter -Wno-implicit-fallthrough"
)
message(
  STATUS
    "⚙️  Setting flags [${SUPRESS_SOME_WARNINGS}] to suppress warnings like: ‘template<class _Tp> struct std::is_pod’ is deprecated: use 'is_standard_layout && is_trivial'"
)
set(ENABLE_SQRT_VECTORIZATION_FLAG "-fno-math-errno")
message(
  STATUS
    "⚙️  Setting flag '${ENABLE_SQRT_VECTORIZATION_FLAG}' to enable vectorization of square root calculations without setting errno on errors"
)
set(DONT_WARN_ABOUT_EXCP_WHILE_DIVIDE_FLAG "-fno-trapping-math")
message(
  STATUS
    "⚙️  Setting flag '${DONT_WARN_ABOUT_EXCP_WHILE_DIVIDE_FLAG}' to avoid warnings about exceptions on divide operations"
)

set(COMMON_CPP_FLAGS
    "-Wall -Wextra -Wpedantic -pthread -O3 -Wmisleading-indentation")
set(FINAL_CPP_FLAGS
    "${COMMON_CPP_FLAGS} ${SUPRESS_SOME_WARNINGS} ${ENABLE_SQRT_VECTORIZATION_FLAG} ${DONT_WARN_ABOUT_EXCP_WHILE_DIVIDE_FLAG}"
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FINAL_CPP_FLAGS}")

# Initialize variables for optional flags
set(ANTI_ALIASING_FLAG "")
set(VECTORIZATION_FLAGS "")

# Checking support of the AVX, AVX2 and AVX-512
execute_process(
  COMMAND lscpu
  OUTPUT_VARIABLE CPU_INFO
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(CPU_INFO MATCHES "avx512")
  set(VECTOR_WIDTH 512)
elseif(CPU_INFO MATCHES "avx" OR CPU_INFO MATCHES "avx2")
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
    add_compile_options(${CLANG_FLAGS} ${VECTORIZATION_FLAGS}
                        ${ANTI_ALIASING_FLAG})
    add_compile_options("-mprefer-vector-width=${VECTOR_WIDTH}")
    message(
      STATUS
        "⚙️  Clang-specific vectorization flags added: [${CLANG_FLAGS} ${VECTORIZATION_FLAGS} ${ANTI_ALIASING_FLAG}]"
    )

    # Check for GCC and set specific flags
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(ANTI_ALIASING_FLAG "-fstrict-aliasing")
    set(VECTORIZATION_FLAGS "-ftree-vectorize")
    set(GCC_FLAGS "-flarge-source-files")
    add_compile_options(${GCC_FLAGS} ${VECTORIZATION_FLAGS}
                        ${ANTI_ALIASING_FLAG})
    add_compile_options("-mprefer-vector-width=${VECTOR_WIDTH}")
    message(
      STATUS
        "⚙️  GCC-specific vectorization flags added: [${GCC_FLAGS} ${VECTORIZATION_FLAGS} ${ANTI_ALIASING_FLAG}]"
    )

    # Check for Intel compiler and set specific flags
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(ANTI_ALIASING_FLAG "-ansi-alias")
    set(VECTORIZATION_FLAGS "-restrict" "-xHost" "-vecabi=cmdtarget"
                            "-qopt-zmm-usage=high")
    set(INTEL_COMPILER_FLAGS "-flarge-source-files")

    add_compile_options(${INTEL_COMPILER_FLAGS} ${VECTORIZATION_FLAGS}
                        ${ANTI_ALIASING_FLAG})
    message(
      STATUS
        "⚙️  Intel-specific vectorization flags added: [${INTEL_COMPILER_FLAGS} ${VECTORIZATION_FLAGS} ${ANTI_ALIASING_FLAG}]"
    )
  endif()
  message(STATUS "⚙️  Vector width set to: ${VECTOR_WIDTH}")
endif()

# Append the common and custom flags to CMAKE_CXX_FLAGS
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FINAL_CPP_FLAGS} -mno-amx-tile")
message(STATUS "⚙️  Compiling with C++ flags: [${CMAKE_CXX_FLAGS}]")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║           FLOATING-POINT ROUNDING MODE CONFIGURATION                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# ╔═► Configure compiler-specific floating-point rounding ◄══════════════════╗
# ║ Detect the compiler being used (Intel, GNU) and apply appropriate        ║ ║
# floating-point rounding flags for strict computation control.            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  add_compile_options(-fp-model strict)
  message(
    STATUS
      "⚙️  Intel: Using strict floating-point rounding with flag: -fp-model strict"
  )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options(-frounding-math)
  message(
    STATUS
      "⚙️  GNU: Using strict floating-point rounding with flag: -frounding-math"
  )

else()
  message(
    WARNING
      "⚠️ Unsupported compiler. You may need to manually set appropriate flags for strict floating-point rounding."
  )
endif()

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║           BUILD TYPE CONFIGURATION                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# ╔═► Check if we're building in Debug or Release mode ◄═════════════════════╗
# ║ Define the appropriate macros (`DEBUG` or `RELEASE`) based on the build  ║ ║
# type (Debug, Release, etc.).                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "🛠️  Compiling in DEBUG mode")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  message(STATUS "🚀 Compiling in RELEASE mode")
else()
  message(WARNING "⚠️ Unknown build type: ${CMAKE_BUILD_TYPE}")
endif()

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║           START DEBUG / RELEASE OPTIONS                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# ╔═► Start-specific Debug and Release options ◄═════════════════════════════╗
# ║ Allow the user to specify custom "Start Debug" or "Start Release" options║ ║
# for specific behavior during compilation.                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
if(START_DEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
  add_compile_definitions(START_DEBUG)
  message(
    STATUS
      "🛠️  Start project compiling mode set to START_DEBUG. The variable CMAKE_BUILD_TYPE is set to \"Debug\".
        For performance reasons, you should set CMAKE_BUILD_TYPE to \"Release\"."
  )
elseif(START_RELEASE)
  add_compile_definitions(START_RELEASE)
  message(STATUS "🚀 Start project compiling mode set to START_RELEASE.")
else()
  message(FATAL_ERROR "⚠️ Neither START_DEBUG nor START_RELEASE is defined.")
endif()
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║              END OF CONFIGURING OPTIONS                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/include/DataHandling
  ${CMAKE_SOURCE_DIR}/include/FiniteElementMethod
  ${CMAKE_SOURCE_DIR}/include/FiniteElementMethod/BoundaryConditions
  ${CMAKE_SOURCE_DIR}/include/FiniteElementMethod/Cell
  ${CMAKE_SOURCE_DIR}/include/FiniteElementMethod/Cubature
  ${CMAKE_SOURCE_DIR}/include/Generators
  ${CMAKE_SOURCE_DIR}/include/Geometry
  ${CMAKE_SOURCE_DIR}/include/Particle
  ${CMAKE_SOURCE_DIR}/include/SessionManagement
  ${CMAKE_SOURCE_DIR}/include/Utilities)

add_subdirectory(src)

set(CPP_SOURCES src/main.cpp src/ModelingMainDriver.cpp)
set(SOURCES ${CPP_SOURCES})
if(GPU_CUDA_ON)
  set(CUDA_SOURCES # !!! Do not forget place .cu files here !!!
  )
  set(SOURCES ${SOURCES} ${CUDA_SOURCES})
endif()

set(EXECUTABLE_NAME nia_start_core)
add_executable(${EXECUTABLE_NAME} ${SOURCES})

set_target_properties(${EXECUTABLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                    "${CMAKE_SOURCE_DIR}")
target_link_libraries(
  ${EXECUTABLE_NAME}
  PRIVATE ${CGAL_NAMESPACE}
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
          FiniteElementMethod_BoundaryConditions
          FiniteElementMethod_Cell
          FiniteElementMethod_Cubature
          FiniteElementMethod_LinearAlgebraManagers
          Generators
          Geometry
          Particle
          SessionManagers
          Utilities
)

if(GPU_CUDA_ON)
  target_compile_options(
    ${EXECUTABLE_NAME} PRIVATE "-mno-amx-bf16" "-mno-amx-int8" "-mno-amx-tile")
  target_link_libraries(${EXECUTABLE_NAME} PRIVATE ${CUDA_LIBRARIES})
  set_target_properties(${EXECUTABLE_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION
                                                      ON)
endif()
if(START_OPENMP_ON)
  target_link_libraries(${EXECUTABLE_NAME} PRIVATE ${OpenMP_LIBRARIES})
endif()

add_custom_target(
  distclean COMMAND rm -rf MakeCache.txt CMakeFiles Makefile
                    cmake_install.cmake CTestTestfile.cmake Testing)
