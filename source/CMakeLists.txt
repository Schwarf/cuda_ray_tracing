cmake_minimum_required(VERSION 3.20)
project(cuda_ray_tracing LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_C_STANDARD 99)
include(CheckCXXCompilerFlag)

function(enable_cxx_compiler_flag_if_supported flag)
    string(FIND "${CMAKE_CUDA_FLAGS}" "${flag}" flag_already_set)
    if (flag_already_set EQUAL -1)
        check_cxx_compiler_flag("${flag}" flag_supported)
        if (flag_supported)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}" PARENT_SCOPE)
        endif ()
        unset(flag_supported CACHE)
    endif ()
endfunction()

enable_cxx_compiler_flag_if_supported("-Wall")
enable_cxx_compiler_flag_if_supported("-Wextra")
enable_cxx_compiler_flag_if_supported("-pedantic")
enable_cxx_compiler_flag_if_supported("-O0")


add_executable(cuda_cast render.cu)
add_executable(test ./cuda_playground/test.cu)
add_executable(simple_pointer cuda_playground/pointer_arithmetic_revisited/simple_c.cu)

add_subdirectory(cuda_implementation)
add_subdirectory(cuda_playground)

target_link_libraries(cuda_cast cuda_misc_lib cuda_interfaces)
set_target_properties(cuda_cast PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


