# -----------------------------------------------------------------------------
# ROOT CMakeLists.txt for cuHSDLP (Unified Build System)
# -----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.24)

# Build the major GPU families supported by the selected CUDA compiler. This
# stays valid when the installed GPU is newer than the toolkit (for example,
# an sm_120 GPU with CUDA 12.4). Release builds may still override the list.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES all-major)
endif()

# Project config
project(cuhsdlp LANGUAGES C CXX CUDA)

set(CUHSDLP_VERSION_MAJOR 0)
set(CUHSDLP_VERSION_MINOR 0)
set(CUHSDLP_VERSION_PATCH 1)

set(CUHSDLP_VERSION
    "${CUHSDLP_VERSION_MAJOR}.${CUHSDLP_VERSION_MINOR}.${CUHSDLP_VERSION_PATCH}")
add_compile_definitions(CUHSDLP_VERSION="${CUHSDLP_VERSION}")

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

# C/C++ standards
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Set default build type to Release if not specified (Optimized builds)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
                                                     "MinSizeRel" "RelWithDebInfo")
endif()

# Global Compile flags (corresponding to CFLAGS/NVCCFLAGS)
add_compile_definitions(CUSPARSE_ENABLE_EXPERIMENTAL_API)
add_compile_options(-O3 -Wall -Wextra -g)
if(NOT WIN32)
  add_compile_options(-fPIC)
endif()

# Windows compatibility
if(WIN32)
  add_definitions(-Dstrtok_r=strtok_s)
endif()

# CUDA standards and RDC (Relocatable Device Code)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------------
# CONTROL OPTIONS
# -----------------------------------------------------------------------------
include(CMakeDependentOption)

option(CUHSDLP_BUILD_STATIC_LIB "Build the cuHSDLP static library" ON)
option(CUHSDLP_BUILD_SHARED_LIB "Build the cuHSDLP shared library" ON)

# format: cmake_dependent_option(OPTION "docstring" DEFAULT_VALUE
# "DEPENDENCY_VARIABLE" FORCE_OFF_VALUE)
cmake_dependent_option(CUHSDLP_BUILD_PYTHON "Build the cuHSDLP Python bindings" OFF
                       "CUHSDLP_BUILD_STATIC_LIB" OFF)

cmake_dependent_option(CUHSDLP_BUILD_CLI "Build the cuHSDLP command-line executable" ON
                       "CUHSDLP_BUILD_STATIC_LIB" OFF)

cmake_dependent_option(CUHSDLP_BUILD_TESTS "Build the cuHSDLP test suite" OFF
                       "CUHSDLP_BUILD_STATIC_LIB" OFF)

# -----------------------------------------------------------------------------
# FIND DEPENDENCIES
# -----------------------------------------------------------------------------
# Core dependencies (required for Julia/Yggdrasil and Python)
find_package(CUDAToolkit REQUIRED)

include(FetchContent)
# 1. Try to find ZLIB in the system first (Standard for Linux/Mac)
find_package(ZLIB QUIET)

if(ZLIB_FOUND)
  message(STATUS "Found System ZLIB: ${ZLIB_LIBRARIES}")
else()
  # 2. If not found (Common on Windows), fetch from source
  message(STATUS "ZLIB not found in system. Fetching from source...")

  FetchContent_Declare(
    zlib
    GIT_REPOSITORY https://github.com/madler/zlib.git
    GIT_TAG v1.3)
  FetchContent_MakeAvailable(zlib)

  # 3. Create alias ZLIB::ZLIB to match your CORE_LINK_LIBS usage
  if(NOT TARGET ZLIB::ZLIB)
    if(TARGET zlibstatic)
      # Prefer zlibstatic to ensure static linking (avoids DLL issues on Windows)
      add_library(ZLIB::ZLIB ALIAS zlibstatic)
      # Expose include directories so zlib.h can be found
      target_include_directories(zlibstatic INTERFACE ${zlib_SOURCE_DIR}
                                                      ${zlib_BINARY_DIR})
    else()
      # Fallback if the target is simply named 'zlib'
      add_library(ZLIB::ZLIB ALIAS zlib)
      target_include_directories(zlib INTERFACE ${zlib_SOURCE_DIR}
                                                ${zlib_BINARY_DIR})
    endif()
  endif()
endif()

if(CUHSDLP_BUILD_PYTHON)
  # Dependencies required only for Python bindings
  find_package(pybind11 CONFIG REQUIRED)
  find_package(Python3 COMPONENTS Interpreter REQUIRED) # For versioning script
                                                        # and pybind11
endif()

set(PSLP_VERSION_TAG "v0.0.11")

FetchContent_Declare(
  pslp
  GIT_REPOSITORY https://github.com/dance858/PSLP.git
  GIT_TAG ${PSLP_VERSION_TAG})

FetchContent_MakeAvailable(pslp)

# -----------------------------------------------------------------------------
# SOURCE DISCOVERY & TARGET DEFINITION
# -----------------------------------------------------------------------------
set(CORE_SOURCES
    src/cuhsdlp.c
    src/mps_parser.c
    src/presolver.c
    src/utils/matrix_formats.c
    src/utils/pid_controller.c
    src/cone_projector.cu
    src/preconditioner.cu
    src/result.cu
    src/setup.cu
    src/solver.cu
    src/spmv_backend.cu
    src/termination_checker.cu
    src/utils/display.cu
    src/utils/power_iteration.cu
    src/utils/utils.cu)

# Set common include directories for the core libraries
set(CORE_INCLUDE_DIRS
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include # Public API headers
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/internal # Internal implementation headers
)

# Set common link libraries
set(CORE_LINK_LIBS PUBLIC
                   CUDA::cudart
                   CUDA::cublas
                   CUDA::cusparse
                   CUDA::curand
                   ZLIB::ZLIB
                   PSLP)

# Compile the CUDA-heavy implementation once and reuse it in both library
# variants. Dependencies remain attached to the final public targets below.
add_library(cuhsdlp_objects OBJECT ${CORE_SOURCES})
target_include_directories(
  cuhsdlp_objects
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/internal ${pslp_SOURCE_DIR}/PSLP)
target_compile_definitions(cuhsdlp_objects PRIVATE PSLP_VERSION=\"${PSLP_VERSION_TAG}\")
target_link_libraries(cuhsdlp_objects PRIVATE CUDA::cudart CUDA::cublas
                                                 CUDA::cusparse CUDA::curand
                                                 ZLIB::ZLIB PSLP)
set_target_properties(
  cuhsdlp_objects
  PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_SEPARABLE_COMPILATION ON)

# -----------------------------------------------------------------------------
# 1. Core STATIC Library (cuhsdlp_core)
# -----------------------------------------------------------------------------
if(CUHSDLP_BUILD_STATIC_LIB)
  add_library(cuhsdlp_core STATIC $<TARGET_OBJECTS:cuhsdlp_objects>)
  target_include_directories(cuhsdlp_core ${CORE_INCLUDE_DIRS})
  target_link_libraries(cuhsdlp_core ${CORE_LINK_LIBS})

  set_target_properties(
    cuhsdlp_core
    PROPERTIES POSITION_INDEPENDENT_CODE ON
               CUDA_SEPARABLE_COMPILATION ON
               CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

# -----------------------------------------------------------------------------
# 2. Shared Library (libcuhsdlp.so)
# -----------------------------------------------------------------------------
if(CUHSDLP_BUILD_SHARED_LIB)
  add_library(cuhsdlp_shared SHARED $<TARGET_OBJECTS:cuhsdlp_objects>)
  target_include_directories(cuhsdlp_shared ${CORE_INCLUDE_DIRS})
  target_link_libraries(cuhsdlp_shared ${CORE_LINK_LIBS})

  # Shared library must resolve device symbols as it is a final link point
  set_target_properties(
    cuhsdlp_shared
    PROPERTIES OUTPUT_NAME "cuhsdlp"
               RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
               CUDA_SEPARABLE_COMPILATION ON
               CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

# -----------------------------------------------------------------------------
# 3. CLI Executable (cuhsdlp)
# -----------------------------------------------------------------------------
if(CUHSDLP_BUILD_CLI)
  if(NOT TARGET cuhsdlp_core)
    message(FATAL_ERROR
            "CUHSDLP_BUILD_CLI=ON requires CUHSDLP_BUILD_STATIC_LIB=ON.")
  endif()

  add_executable(cuhsdlp_cli src/cli.c)

  target_include_directories(cuhsdlp_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                 ${CMAKE_CURRENT_SOURCE_DIR}/internal)

  # Link CLI to the static core library
  target_link_libraries(cuhsdlp_cli PRIVATE cuhsdlp_core)

  # CLI is a final executable, it must resolve device symbols
  set_target_properties(
    cuhsdlp_cli
    PROPERTIES OUTPUT_NAME "cuhsdlp"
               RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
               CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

# -----------------------------------------------------------------------------
# 4. Tests (CTest Integration)
# -----------------------------------------------------------------------------
if(CUHSDLP_BUILD_TESTS)
  if(NOT TARGET cuhsdlp_core)
    message(FATAL_ERROR
            "CUHSDLP_BUILD_TESTS=ON requires CUHSDLP_BUILD_STATIC_LIB=ON.")
  endif()

  enable_testing()
  file(GLOB TEST_SOURCES CONFIGURE_DEPENDS
       "${CMAKE_CURRENT_SOURCE_DIR}/test/unit/*.c"
       "${CMAKE_CURRENT_SOURCE_DIR}/test/unit/*.cu"
       "${CMAKE_CURRENT_SOURCE_DIR}/test/integration/*.c"
       "${CMAKE_CURRENT_SOURCE_DIR}/test/integration/*.cu")

  foreach(TEST_SRC ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)

    add_executable(${TEST_NAME} ${TEST_SRC})

    # Link tests to the core static library
    target_link_libraries(${TEST_NAME} PRIVATE cuhsdlp_core)

    # Set up test includes
    target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                   PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/internal)

    # Tests are final executables, they must resolve device symbols
    set_target_properties(
      ${TEST_NAME}
      PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tests"
                 CUDA_RESOLVE_DEVICE_SYMBOLS ON)

    # Register with CTest
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
  endforeach()
endif()

# -----------------------------------------------------------------------------
# 5. Python Bindings (Conditional)
# -----------------------------------------------------------------------------
if(CUHSDLP_BUILD_PYTHON)
  if(NOT TARGET cuhsdlp_core)
    message(FATAL_ERROR
            "CUHSDLP_BUILD_PYTHON=ON requires CUHSDLP_BUILD_STATIC_LIB=ON.")
  endif()

  add_subdirectory(python_bindings)
endif()

# -----------------------------------------------------------------------------
# 6. Install Targets
# -----------------------------------------------------------------------------

if(CUHSDLP_BUILD_PYTHON)
  install(DIRECTORY include/ DESTINATION include/ FILES_MATCHING PATTERN "*.h")
else()
  if(TARGET cuhsdlp_core)
    install(TARGETS cuhsdlp_core ARCHIVE DESTINATION lib)
  endif()

  if(TARGET cuhsdlp_shared)
    install(
      TARGETS cuhsdlp_shared
      LIBRARY DESTINATION lib
      RUNTIME DESTINATION bin) # 'bin' for DLLs on Windows, 'lib' for .so on Linux
  endif()

  if(TARGET cuhsdlp_cli)
    install(TARGETS cuhsdlp_cli RUNTIME DESTINATION bin)
  endif()

  install(DIRECTORY include/ DESTINATION include/ FILES_MATCHING PATTERN "*.h")
endif()
