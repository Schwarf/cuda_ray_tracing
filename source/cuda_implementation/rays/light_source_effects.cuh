//
// Created by andreas on 22.12.21.
//

#ifndef LIGHT_SOURCE_EFFECTS_CUH
#define LIGHT_SOURCE_EFFECTS_CUH
#include "interfaces/i_light_source_effects.cuh"
#include "interfaces/i_light_source.cuh"
#include "interfaces/i_hit_record.cuh"
#include "interfaces/i_ray.cuh"
#include "ray_interactions.cuh"
#include "ray.cuh"
#include "hit_record.cuh"
#include "light_source.cuh"

class LightSourceEffects : public ILightSourceEffects
{
public:
	__device__ __host__ LightSourceEffects(ILightSource** &list, IObjectList** & object_list, size_t size);
	__device__ __host__ void compute_light_source_effects(const IRay &ray,
											  IHitRecord &hit_record,
											  float &diffuse_intensity,
											  float &specular_intensity) const final;

private:
	ILightSource ** light_source_list_;
	IObjectList ** object_list_;
	size_t number_of_light_sources_;
};


#endif //LIGHT_SOURCE_EFFECTS_CUH
