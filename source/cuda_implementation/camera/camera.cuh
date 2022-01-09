//
// Created by andreas on 09.01.22.
//

#ifndef CAMERA_CUH
#define CAMERA_CUH
#include "interfaces/i_camera.cuh"


class Camera: public ICamera
{
public:
	__device__ __host__ Camera(int image_width, int image_height, float viewport_width, float focal_length);
	__device__ __host__ void get_pixel_coordinates(const size_t &width_index,
												   const size_t &height_index,
												   float &u,
												   float &v) const override;
	__device__ __host__ Color get_pixel_color(Ray &ray,
											  IObjectList ** object_list,
											  ILightSourceEffects **light_source_effects) override;
	__device__ __host__ Color get_refracted_color(const Ray &incoming_ray,
												  const HitRecord &incoming_hit_record,
												  IObjectList ** object_list,
												  ILightSourceEffects **light_source_effects) override;
	__device__ __host__ Color get_reflected_color(const Ray &incoming_ray,
												  const IHitRecord &incoming_hit_record,
												  IObjectList ** object_list,
												  ILightSourceEffects **light_source_effects) override;
	__device__ __host__ ~Camera() = default;

private:
	int image_width_{};
	int image_height_{};
	float focal_length_{};
	float aspect_ratio_{};
	Point3D origin_{0., 0., 0.};
	Vector3D horizontal_direction_{0., 0., 0.};
	Vector3D vertical_direction_{0., 0., 0.};
	Point3D lower_left_corner_{0., 0., 0.};

};


#endif //CAMERA_CUH
