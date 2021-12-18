//
// Created by andreas on 18.12.21.
//

#ifndef RAY_INTERACTIONS_CUH
#define RAY_INTERACTIONS_CUH
#include "interfaces/i_ray_interactions.cuh"
#include "interfaces/i_ray.cuh"
#include "interfaces/i_hit_record.cuh"
#include "ray.cuh"

class RayInteractions final
{

public:
	__device__ __host__ virtual void diffuse_scatter(const IHitRecord &hit_record, IRay & scattered_ray) = 0;
	__device__ __host__ virtual void specular_scatter(const IRay & incoming_ray, const IHitRecord & hit_record, IRay & scattered_ray) = 0;
	__device__ __host__ virtual void refraction_scatter(const IRay & incoming_ray, const IHitRecord & hit_record, IRay & scattered_ray) = 0;

};


#endif //RAY_INTERACTIONS_CUH
