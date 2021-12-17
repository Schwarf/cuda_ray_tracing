set(CMAKE_CUDA_STANDARD 14)

add_library(cuda_misc_lib STATIC miscellaneous/templates/c_vector.cuh  rays/ray.cuh rays/ray.cu objects/sphere.cuh objects/sphere.cu
            materials/material.cu materials/material.cuh)
add_library(cuda_interfaces INTERFACE rays/interfaces/i_ray.cuh objects/interfaces/i_sphere.cuh materials/interfaces/i_material.cuh)

set_target_properties(cuda_misc_lib PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
