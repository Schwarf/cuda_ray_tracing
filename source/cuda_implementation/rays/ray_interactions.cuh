//
// Created by andreas on 18.12.21.
//

#ifndef RAY_INTERACTIONS_CUH
#define RAY_INTERACTIONS_CUH
#include "interfaces/i_ray.cuh"
#include "interfaces/i_hit_record.cuh"
#include "ray.cuh"

__device__ __host__ inline void diffuse_scatter(const IHitRecord &hit_record, IRay & scattered_ray)
{
	scattered_ray.set_origin(hit_record.hit_point());
	scattered_ray.set_direction(hit_record.hit_normal());
}

__device__ __host__ inline void specular_scatter(const IRay &incoming_ray, const IHitRecord &hit_record,  IRay & scattered_ray)
{

	auto reflection = [](const Vector3D &v, const Vector3D &n)
	{
		return v - n * (2.f * (v * n));
	};
	auto const reflected_direction = reflection(incoming_ray.direction_normalized(), hit_record.hit_normal());
	scattered_ray.set_origin(hit_record.hit_point());
	scattered_ray.set_direction(reflected_direction);
}

#endif //RAY_INTERACTIONS_CUH
