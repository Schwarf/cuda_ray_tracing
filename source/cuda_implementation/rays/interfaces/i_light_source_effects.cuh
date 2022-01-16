//
// Created by andreas on 22.12.21.
//

#ifndef I_LIGHT_SOURCE_LIST_H
#define I_LIGHT_SOURCE_LIST_H
#include "./../../rays/interfaces/i_ray.cuh"
#include "./../../rays/interfaces/i_hit_record.cuh"
#include "./../../objects/interfaces/i_object_list.cuh"
class ILightSourceEffects
{
public:
	__device__ __host__	virtual void compute_light_source_effects(const IRay &ray,
																	 const IHitRecord &hit_record,
																	 float &diffuse_intensity,
																	 float &specular_intensity) const = 0;
	__device__ __host__ virtual void set_background_colors(const Color & background_color1, const Color & background_color2) = 0;
	__device__ __host__ virtual Color get_background(float parameter) const = 0;
};

#endif //I_LIGHT_SOURCE_LIST_H
