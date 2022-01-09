//
// Created by andreas on 09.01.22.
//

#include "camera.cuh"
__device__ __host__ Camera::Camera(int image_width, int image_height, float viewport_width, float focal_length)
{
	image_width_ = image_width;
	image_height_ = image_height;
	focal_length_ = focal_length;
	aspect_ratio_ = static_cast<float>(image_width) / static_cast<float>(image_height);
	float viewport_height = viewport_width / aspect_ratio_;
	horizontal_direction_[0] = viewport_width;
	vertical_direction_[1] = viewport_height;

}


__device__ __host__ void Camera::get_pixel_coordinates(const size_t &width_index,
													   const size_t &height_index,
													   float &u,
													   float &v) const
{

	u = float(width_index)  / float(image_width_ - 1);
	v = float(height_index)  / float(image_height_ - 1);
}

__device__ __host__ Color Camera::get_pixel_color(Ray &ray,
												  IObjectList **const object_list,
												  ILightSourceEffects **light_source_effects)
{
	Color final_color{0., 0., 0.};
	Color white = Color{1, 1, 1};
	auto hit_record = HitRecord();
	auto reflected_ray = Ray();
	auto is_hit = (*object_list)->any_object_hit_by_ray(ray, hit_record);
	if (!is_hit) {
		return Color{0.2, 0.7, 0.8};
	}
	float diffuse_intensity{};
	float specular_intensity{};
	(*light_source_effects)
		->compute_light_source_effects(ray, hit_record, diffuse_intensity, specular_intensity);
	Color diffuse_color = diffuse_intensity * hit_record.get_material()->diffuse_reflection()
		* hit_record.get_material()->rgb_color();
	Color specular_color = specular_intensity * white * hit_record.get_material()->specular_reflection();
	Color ambient_color = hit_record.get_material()->ambient_reflection() * hit_record.get_material()->rgb_color();
	auto initial_color = diffuse_color + specular_color + ambient_color;

	final_color = initial_color + get_reflected_color(ray, hit_record, object_list, light_source_effects)
		+get_refracted_color(ray, hit_record, object_list, light_source_effects);
	return final_color;

}
__device__ __host__ Color Camera::get_refracted_color(const Ray &incoming_ray,
													  const HitRecord &incoming_hit_record,
													  IObjectList **const object_list,
													  ILightSourceEffects **light_source_effects)
{
	Color refracted_color{0, 0, 0};
	Color white = Color{1, 1, 1};
	auto hit_record = HitRecord();
	hit_record.set_hit_normal(incoming_hit_record.hit_normal());
	hit_record.set_hit_point(incoming_hit_record.hit_point());
	hit_record.set_material(incoming_hit_record.get_material());
	auto ray = incoming_ray;
	auto refracted_ray = Ray();
	for (int i = 0; i < 2; ++i) {
		refraction(ray, hit_record, refracted_ray);
		auto is_refracted_hit = (*object_list)->any_object_hit_by_ray(refracted_ray, hit_record);
		if (!is_refracted_hit) {
			continue;
		}
		float diffuse_refracted_intensity{};
		float specular_refracted_intensity{};
		(*light_source_effects)
			->compute_light_source_effects(ray,
										   hit_record,
										   diffuse_refracted_intensity,
										   specular_refracted_intensity);
		Color diffuse_refracted_color =
			diffuse_refracted_intensity * hit_record.get_material()->diffuse_reflection()
				* hit_record.get_material()->rgb_color();
		Color specular_refracted_color =
			specular_refracted_intensity * white * hit_record.get_material()->specular_reflection();
		Color ambient_refracted_color = hit_record.get_material()->ambient_reflection()
			* hit_record.get_material()->rgb_color();
		refracted_color += diffuse_refracted_color + specular_refracted_color + ambient_refracted_color;
		ray = refracted_ray;
	}
	refracted_color *= hit_record.get_material()->transparency();
	return refracted_color;
}

__device__ __host__ Color Camera::get_reflected_color(const Ray &incoming_ray,
													  const IHitRecord &incoming_hit_record,
													  IObjectList **const object_list,
													  ILightSourceEffects **light_source_effects)
{
	Color reflected_color{0, 0, 0};
	Color white = Color{1, 1, 1};

	Ray ray = incoming_ray;
	auto reflected_ray = Ray();
	auto hit_record = HitRecord();
	hit_record.set_hit_normal(incoming_hit_record.hit_normal());
	hit_record.set_hit_point(incoming_hit_record.hit_point());
	hit_record.set_material(incoming_hit_record.get_material());
	for (int i = 0; i < 2; ++i) {
		specular_scatter(ray, hit_record, reflected_ray);
		auto is_reflected_hit = (*object_list)->any_object_hit_by_ray(reflected_ray, hit_record);
		if (!is_reflected_hit) {
			continue;
		}
		//printf( "Reflected hit  %d \n ", is_reflected_hit);
		//printf( "Reflected hit  %f, %f, %f \n", hit_record.hit_normal()[0], hit_record.hit_normal()[1], hit_record.hit_normal()[2]);
		float diffuse_reflected_intensity{};
		float specular_reflected_intensity{};
		(*light_source_effects)
			->compute_light_source_effects(ray,
										   hit_record,
										   diffuse_reflected_intensity,
										   specular_reflected_intensity);
		Color diffuse_reflected_color =
			diffuse_reflected_intensity * hit_record.get_material()->diffuse_reflection()
				* hit_record.get_material()->rgb_color();
		Color specular_reflected_color =
			specular_reflected_intensity * white * hit_record.get_material()->specular_reflection();
		Color ambient_reflected_color = hit_record.get_material()->ambient_reflection()
			* hit_record.get_material()->rgb_color();
		reflected_color += diffuse_reflected_color + specular_reflected_color + ambient_reflected_color;
		ray = reflected_ray;
	}
	reflected_color *= hit_record.get_material()->ambient_reflection();
	return reflected_color;

}
