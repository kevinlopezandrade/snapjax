cmake_minimum_required(VERSION 3.15...3.27)
project(SppMatmul LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 20)

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Check for CUDA
find_package(CUDAToolkit REQUIRED)
include(CheckLanguage)
check_language(CUDA)

# Include CUDA Headers.
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set(CMAKE_CUDA_FLAGS "-std=c++20 -lcusparse")

set(CMAKE_CXX_FLAGS "-std=c++2a")
set(CMAKE_CXX_FLAGS "-I$ENV{CONDA_PREFIX}/include")

nanobind_add_module(sp_ops NOMINSIZE sp_ops.cc kernel.cc.cu)

target_link_libraries(sp_ops PRIVATE CUDA::cusparse)
