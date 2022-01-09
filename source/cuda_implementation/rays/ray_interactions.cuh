//
// Created by andreas on 18.12.21.
//

#ifndef RAY_INTERACTIONS_CUH
#define RAY_INTERACTIONS_CUH
#include "interfaces/i_ray.cuh"
#include "interfaces/i_hit_record.cuh"
#include "ray.cuh"

__device__ __host__ void diffuse_scatter(const IHitRecord &hit_record, IRay & scattered_ray);
__device__ __host__ void specular_scatter(const IRay &incoming_ray, const IHitRecord &incoming_hit_record, IRay & scattered_ray);
__device__ __host__ void refraction(const IRay &incoming_ray, const IHitRecord &hit_record,  IRay & scattered_ray);

#endif //RAY_INTERACTIONS_CUH
