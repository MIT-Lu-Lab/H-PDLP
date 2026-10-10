# -----------------------------------------------------------------------------
# ROOT CMakeLists.txt for H-PDLP (Unified Build System)
# -----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.24)

# Build the major GPU families supported by the selected CUDA compiler. This
# stays valid when the installed GPU is newer than the toolkit (for example,
# an sm_120 GPU with CUDA 12.4). Release builds may still override the list.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES all-major)
endif()

# Project config
project(hpdlp VERSION 0.0.1 LANGUAGES C CXX CUDA)

add_compile_definitions(HPDLP_VERSION="${PROJECT_VERSION}")

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

option(HPDLP_BUILD_STATIC_LIB "Build the H-PDLP static library" ON)
option(HPDLP_BUILD_SHARED_LIB "Build the H-PDLP shared library" ON)

# format: cmake_dependent_option(OPTION "docstring" DEFAULT_VALUE
# "DEPENDENCY_VARIABLE" FORCE_OFF_VALUE)
cmake_dependent_option(HPDLP_BUILD_PYTHON "Build the H-PDLP Python bindings" OFF
                       "HPDLP_BUILD_STATIC_LIB" OFF)

cmake_dependent_option(HPDLP_BUILD_CLI "Build the H-PDLP command-line executable" ON
                       "HPDLP_BUILD_STATIC_LIB" OFF)

cmake_dependent_option(HPDLP_BUILD_TESTS "Build the H-PDLP test suite" OFF
                       "HPDLP_BUILD_STATIC_LIB" OFF)

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

if(HPDLP_BUILD_PYTHON)
  # Dependencies required only for Python bindings
  set(PYBIND11_FINDPYTHON ON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
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
    src/hpdlp.c
    src/parameters.c
    src/presolver.c
    src/utils/matrix_formats.c
    src/cone_projector.cu
    src/preconditioner.cu
    src/result.cu
    src/setup.cu
    src/solver.cu
    src/spmv_backend.cu
    src/termination_checker.cu
    src/step_control/active_set_boost.cu
    src/step_control/pid_controller.c
    src/step_control/power_iteration.cu
    src/utils/cuda_utils.cu
    src/utils/display.c
    src/utils/error_handling.c
    src/utils/mps_parser.c
    src/utils/utils.c)

# Set common link libraries
set(CORE_LINK_LIBS CUDA::cudart
                   CUDA::cublas
                   CUDA::cusparse
                   CUDA::curand
                   ZLIB::ZLIB
                   PSLP)

# Compile the CUDA-heavy implementation once and reuse it in both library
# variants. Dependencies remain attached to the final public targets below.
add_library(hpdlp_objects OBJECT ${CORE_SOURCES})
target_include_directories(
  hpdlp_objects
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/internal ${pslp_SOURCE_DIR}/PSLP)
target_compile_definitions(hpdlp_objects PRIVATE PSLP_VERSION=\"${PSLP_VERSION_TAG}\")
target_link_libraries(hpdlp_objects PRIVATE ${CORE_LINK_LIBS})
set_target_properties(
  hpdlp_objects
  PROPERTIES POSITION_INDEPENDENT_CODE ON
             CUDA_SEPARABLE_COMPILATION ON
             C_VISIBILITY_PRESET hidden
             CUDA_VISIBILITY_PRESET hidden)

# -----------------------------------------------------------------------------
# 1. Core STATIC Library (hpdlp_core)
# -----------------------------------------------------------------------------
if(HPDLP_BUILD_STATIC_LIB)
  add_library(hpdlp_core STATIC $<TARGET_OBJECTS:hpdlp_objects>)
  target_include_directories(hpdlp_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(hpdlp_core PUBLIC ${CORE_LINK_LIBS})

  set_target_properties(
    hpdlp_core
    PROPERTIES POSITION_INDEPENDENT_CODE ON
               CUDA_SEPARABLE_COMPILATION ON
               CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

# -----------------------------------------------------------------------------
# 2. Shared Library (libhpdlp.so)
# -----------------------------------------------------------------------------
if(HPDLP_BUILD_SHARED_LIB)
  add_library(hpdlp_shared SHARED $<TARGET_OBJECTS:hpdlp_objects>)
  target_include_directories(hpdlp_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(hpdlp_shared PUBLIC ${CORE_LINK_LIBS})

  # Shared library must resolve device symbols as it is a final link point
  set_target_properties(
    hpdlp_shared
    PROPERTIES OUTPUT_NAME "hpdlp"
               RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
               CUDA_SEPARABLE_COMPILATION ON
               CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

# -----------------------------------------------------------------------------
# 3. CLI Executable (hpdlp)
# -----------------------------------------------------------------------------
if(HPDLP_BUILD_CLI)
  add_executable(hpdlp_cli src/cli.c)

  target_include_directories(hpdlp_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                 ${CMAKE_CURRENT_SOURCE_DIR}/internal)

  # Link CLI to the static core library
  target_link_libraries(hpdlp_cli PRIVATE hpdlp_core)

  # CLI is a final executable, it must resolve device symbols
  set_target_properties(
    hpdlp_cli
    PROPERTIES OUTPUT_NAME "hpdlp"
               RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
               CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

# -----------------------------------------------------------------------------
# 4. Tests (CTest Integration)
# -----------------------------------------------------------------------------
if(HPDLP_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

# -----------------------------------------------------------------------------
# 5. Python Bindings (Conditional)
# -----------------------------------------------------------------------------
if(HPDLP_BUILD_PYTHON)
  add_subdirectory(python_bindings)
endif()

# -----------------------------------------------------------------------------
# 6. Install Targets
# -----------------------------------------------------------------------------

if(NOT HPDLP_BUILD_PYTHON)
  if(TARGET hpdlp_core)
    install(TARGETS hpdlp_core ARCHIVE DESTINATION lib)
  endif()

  if(TARGET hpdlp_shared)
    install(
      TARGETS hpdlp_shared
      LIBRARY DESTINATION lib
      RUNTIME DESTINATION bin) # 'bin' for DLLs on Windows, 'lib' for .so on Linux
  endif()

  if(TARGET hpdlp_cli)
    install(TARGETS hpdlp_cli RUNTIME DESTINATION bin)
  endif()

  install(DIRECTORY include/ DESTINATION include/ FILES_MATCHING PATTERN "*.h")
endif()
