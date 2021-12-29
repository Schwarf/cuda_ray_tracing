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


__device__ __host__ inline void refraction(const IRay &incoming_ray, const IHitRecord &hit_record,  IRay & scattered_ray)
{

	float air_refraction_index =1.f;
	auto hit_normal = hit_record.hit_normal();
	float cosine = -fmaxf(-1.f, fminf(1.f, incoming_ray.direction_normalized()*hit_normal));
	float material_refraction_index = hit_record.get_material()->refraction_index();
	if(cosine < 0) {
		// ray is inside sphere, switch refraction_indices and normal
		hit_normal = -1.f*hit_normal;
		float help = material_refraction_index;
		material_refraction_index = air_refraction_index;
		air_refraction_index = help;
	}
	float ratio = air_refraction_index/material_refraction_index;
	float k = 1.f - ratio*ratio*(1.f - cosine*cosine);
	if (k > 0) {
		Vector3D refracted_ray_direction = incoming_ray.direction_normalized()*ratio + hit_normal*(ratio*cosine - sqrtf(k));
		scattered_ray.set_direction(refracted_ray_direction);
		scattered_ray.set_origin(hit_record.hit_point());
	}

}


#endif //RAY_INTERACTIONS_CUH
