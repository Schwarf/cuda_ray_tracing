//
// Created by andreas on 09.01.22.
//

#ifndef CAMERA_CUH
#define CAMERA_CUH
#include "interfaces/i_camera.cuh"


class Camera: public ICamera
{
public:

	__device__ __host__ Camera(size_t image_width, size_t image_height, float viewport_width, float focal_length);
	__device__ __host__ Color get_pixel_color(Ray &ray,
											  IObjectList ** object_list,
											  ILightSourceEffects **light_source_effects) final;
	__device__ __host__ Color get_refracted_color(const Ray &incoming_ray,
												  const HitRecord &incoming_hit_record,
												  IObjectList ** object_list,
												  ILightSourceEffects **light_source_effects) final;
	__device__ __host__ Color get_reflected_color(const Ray &incoming_ray,
												  const IHitRecord &incoming_hit_record,
												  IObjectList ** object_list,
												  ILightSourceEffects **light_source_effects) final;
	__device__ __host__ ~Camera() = default;
	__device__ __host__ Vector3D get_ray_direction(const size_t &width_index,
												   const size_t &height_index) const final;

private:
	size_t image_width_{};
	size_t image_height_{};
	float focal_length_{};
	float aspect_ratio_{};
	Point3D origin_{0., 0., 0.};
	Vector3D horizontal_direction_{0., 0., 0.};
	Vector3D vertical_direction_{0., 0., 0.};
	Point3D lower_left_corner_{0., 0., 0.};
	__device__ __host__ void get_pixel_coordinates(const size_t &width_index,
												   const size_t &height_index,
												   float &u,
												   float &v) const final;
};


#endif //CAMERA_CUH
