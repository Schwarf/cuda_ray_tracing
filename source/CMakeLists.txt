cmake_minimum_required(VERSION 3.20)
project(simple_ray_tracing LANGUAGES CUDA)

set(CMAKE_CUDA_STANDARD 14)
include(CheckCXXCompilerFlag)


add_executable(cuda_cast render.cu)
add_executable(test ./cuda_playground/test.cu)

add_subdirectory(cuda_implementation)
add_subdirectory(cuda_playground)

target_link_libraries(cuda_cast cuda_misc_lib cuda_interfaces)
set_target_properties(cuda_cast PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


