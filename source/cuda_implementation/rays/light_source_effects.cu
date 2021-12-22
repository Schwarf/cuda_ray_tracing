//
// Created by andreas on 22.12.21.
//

#include "light_source_effects.cuh"

__device__ __host__ LightSourceEffects::LightSourceEffects(ILightSource **&list, size_t size):
light_source_list_(list),
number_of_light_sources_(size)
{
}


__device__ __host__ void LightSourceEffects::compute_light_source_effects(const IRay &ray,
													  IHitRecord &hit_record,
													  float &diffuse_intensity,
													  float &specular_intensity) const
{

	auto hit_normal = hit_record.hit_normal();
	auto hit_point = hit_record.hit_point();
	auto material = hit_record.get_material();
	auto light_source_ray = Ray();
	auto shadow_hit_record = HitRecord();
	for (size_t ls_index = 0; ls_index < number_of_light_sources_; ++ls_index) {
		ILightSource *light_source = light_source_list_[ls_index];
		Vector3D light_direction = (light_source->position() - hit_point).normalize();
		light_source_ray.set_direction(light_direction);
		light_source_ray.set_origin(hit_point);
		auto reflected_ray = Ray();
		specular_scatter(light_source_ray, hit_record, reflected_ray);
		diffuse_intensity += light_source->intensity() * fmaxf(0.f, light_direction * hit_normal);
		specular_intensity +=
			powf(fmaxf(0.f, reflected_ray.direction_normalized() * ray.direction_normalized()), material->shininess())
				* light_source->intensity();
	}

}