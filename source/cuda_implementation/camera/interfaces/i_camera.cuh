//
// Created by andreas on 09.01.22.
//

#ifndef I_CAMERA_CUH
#define I_CAMERA_CUH
#include "./../../miscellaneous/templates/n_tuple.cuh"
#include "./../../rays/ray.cuh"
#include "./../../rays/hit_record.cuh"
#include "./../../objects/interfaces/i_object_list.cuh"
#include "./../../rays/interfaces/i_light_source_effects.cuh"
#include "./../../rays/interfaces/i_light_source.cuh"
#include "./../../rays/ray_interactions.cuh"

class ICamera
{
public:
	__device__ __host__ virtual Color get_pixel_color(Ray &ray,
													  IObjectList ** object_list,
													  ILightSourceEffects **light_source_effects) = 0;

	__device__ __host__ virtual Color get_refracted_color(const Ray &incoming_ray, const HitRecord &incoming_hit_record,
														  IObjectList ** object_list,
														  ILightSourceEffects **light_source_effects) = 0;
	__device__ __host__ virtual Color get_reflected_color(const Ray &incoming_ray,
														  const IHitRecord &incoming_hit_record,
														  IObjectList ** object_list,
														  ILightSourceEffects **light_source_effects) = 0;

	__device__ __host__ virtual Vector3D get_ray_direction(const size_t &width_index,
												  const size_t &height_index) const = 0;
	__device__ __host__ ~ICamera() = default;

private:
	__device__ __host__ virtual void get_pixel_coordinates(const size_t &width_index,
														   const size_t &height_index,
														   float &u,
														   float &v) const = 0;

};
#endif //I_CAMERA_CUH
