//
// Created by andreas on 18.12.21.
//

#ifndef I_RAY_INTERACTIONS_CUH
#define I_RAY_INTERACTIONS_CUH

#include "i_ray.cuh"
#include "i_hit_record.cuh"

class IRayInteractions
{
public:
	__device__ __host__ virtual void diffuse_scatter(const IHitRecord &hit_record, IRay & scattered_ray) = 0;
	__device__ __host__ virtual void specular_scatter(const IRay & incoming_ray, const IHitRecord & hit_record, IRay & scattered_ray) = 0;
	__device__ __host__ virtual void refraction_scatter(const IRay & incoming_ray, const IHitRecord & hit_record, IRay & scattered_ray) = 0;
};
#endif // I_RAY_INTERACTIONS_CUH
