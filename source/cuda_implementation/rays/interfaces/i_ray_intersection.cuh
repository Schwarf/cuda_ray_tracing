//
// Created by andreas on 03.10.21.
//

#ifndef CUDA_RAY_TRACING_I_RAY_INTERSECTION_H
#define CUDA_RAY_TRACING_I_RAY_INTERSECTION_H
#include "i_ray.cuh"
#include "i_hit_record.cuh"

class IRayIntersection
{
public:
	__device__ __host__ virtual bool does_ray_intersect(const IRay &ray,  IHitRecord & hit_record) const = 0;
};

#endif //CUDA_RAY_TRACING_I_RAY_INTERSECTION_H
