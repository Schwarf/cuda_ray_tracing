set(CMAKE_CUDA_STANDARD 14)

add_library(cuda_misc_lib STATIC miscellaneous/templates/n_tuple.cuh rays/ray.cuh rays/ray.cu objects/sphere.cuh objects/sphere.cu
            materials/material.cu materials/material.cuh rays/hit_record.cu rays/hit_record.cuh rays/ray_interactions.cuh rays/ray_interactions.cu rays/light_source.cu rays/light_source.cuh
            objects/object_list.cuh miscellaneous/CUDAMemory.cuh objects/object_list.cu create_scene/create_scene.cuh rays/light_source_effects.cu rays/light_source_effects.cuh
            camera/camera.cu camera/camera.cuh)
add_library(cuda_interfaces INTERFACE rays/interfaces/i_ray.cuh objects/interfaces/i_sphere.cuh materials/interfaces/i_material.cuh
        rays/interfaces/i_hit_record.cuh rays/interfaces/i_light_source.cuh objects/interfaces/i_object_list.cuh rays/interfaces/i_light_source_effects.cuh
        camera/interfaces/i_camera.cuh)

set_target_properties(cuda_misc_lib PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
