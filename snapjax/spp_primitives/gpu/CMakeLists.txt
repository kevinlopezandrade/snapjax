cmake_minimum_required(VERSION 3.30)
# Include native code for Hopper and Blackwell, plus forward-compatible PTX.
# Explicit targets also allow building on a CPU-only image builder.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "90-real;100-real;100-virtual")
endif()
project(SnapJaxCUDA LANGUAGES CXX CUDA)

find_package(Python 3.13 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(CUDAToolkit 12.8 REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE NB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -c "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_VARIABLE XLA_INCLUDE OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)

nanobind_add_module(snapjax_cuda NOMINSIZE sp_ops.cc kernel.cc.cu)
target_include_directories(snapjax_cuda PRIVATE "${XLA_INCLUDE}")
target_link_libraries(snapjax_cuda PRIVATE CUDA::cudart_static)
set_target_properties(snapjax_cuda PROPERTIES
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
  CUDA_RUNTIME_LIBRARY Static)
install(TARGETS snapjax_cuda LIBRARY DESTINATION .)
