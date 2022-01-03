//
// Created by andreas on 01.11.21.
//

#include "cuda_implementation/miscellaneous/templates/n_tuple.cuh"
#include "cuda_implementation/rays/ray.cuh"
#include "cuda_implementation/rays/hit_record.cuh"
#include "cuda_implementation/miscellaneous/CUDAMemory.cuh"
#include "cuda_implementation/objects/interfaces/i_object_list.cuh"
#include "cuda_implementation/create_scene/create_scene.cuh"
#include "cuda_implementation/rays/interfaces/i_light_source_effects.cuh"
#include "cuda_implementation/rays/interfaces/i_light_source.cuh"
#include "cuda_implementation/rays/ray_interactions.cuh"
#include <fstream>
#include <iostream>


__device__ __host__ Color get_reflected_color(const Ray &incoming_ray, IObjectList **const object_list,
											  ILightSourceEffects **light_source_effects)
{
	Color reflected_color{0, 0, 0};
	Color white = Color{1, 1, 1};

	auto ray = incoming_ray;
	auto reflected_ray = Ray();
	auto hit_reflected_record = HitRecord();
	for (int i = 0; i < 2; ++i) {
		specular_scatter(ray, hit_reflected_record, reflected_ray);
		auto is_reflected_hit = (*object_list)->any_object_hit_by_ray(reflected_ray, hit_reflected_record);
		if (!is_reflected_hit) {
			continue;
		}
		float diffuse_reflected_intensity{};
		float specular_reflected_intensity{};
		(*light_source_effects)
			->compute_light_source_effects(ray,
										   hit_reflected_record,
										   diffuse_reflected_intensity,
										   specular_reflected_intensity);
		Color diffuse_reflected_color =
			diffuse_reflected_intensity * hit_reflected_record.get_material()->diffuse_reflection()
				* hit_reflected_record.get_material()->rgb_color();
		Color specular_reflected_color =
			specular_reflected_intensity * white * hit_reflected_record.get_material()->specular_reflection();
		Color ambient_reflected_color = hit_reflected_record.get_material()->ambient_reflection()
			* hit_reflected_record.get_material()->rgb_color();
		reflected_color += diffuse_reflected_color + specular_reflected_color + ambient_reflected_color;
		ray = reflected_ray;
	}
	reflected_color *= hit_reflected_record.get_material()->ambient_reflection();
	return reflected_color;
}

__device__ __host__ Color get_pixel_color(Ray &ray,
										  IObjectList **const object_list,
										  ILightSourceEffects **light_source_effects,
										  size_t recursion_depth)
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

	final_color = initial_color + get_reflected_color(ray, object_list, light_source_effects);
	return final_color;
}

__global__ void render_it(Vector3D *buffer, size_t max_width, size_t max_height, IObjectList **object_list,
						  ILightSourceEffects **light_source_effects)
{
	size_t width = threadIdx.x + blockIdx.x * blockDim.x;
	size_t height = threadIdx.y + blockIdx.y * blockDim.y;

	//size_t width = threadIdx.x;
	//size_t height = blockIdx.x;
	if ((width >= max_width) || (height >= max_height)) {
		return;
	}
	float x_direction = float(width) - float(max_width) / 2.f;
	float y_direction = float(height) - float(max_height) / 2.f;
	float z_direction = -float(max_height + max_width) / 2.f;

	Vector3D direction = Vector3D{x_direction, y_direction, z_direction}.normalize();
	Point3D origin = Point3D{0, 0, 0};
	auto ray = Ray(origin, direction);
	Color pixel_color = get_pixel_color(ray, object_list, light_source_effects, 2);
	size_t pixel_index = height * max_width + width;
	buffer[pixel_index] = pixel_color;

}

int main()
{
	constexpr size_t width = 1024;
	constexpr size_t height = 768;
//	constexpr size_t number_of_blocks = 768;
//	constexpr size_t number_of_threads = 512;
	constexpr size_t number_of_objects = 5;
	constexpr size_t number_of_light_sources = 3;
	// Why is 32 the maximum number of threads per block
	constexpr size_t threads_per_block = 16;
	dim3 number_of_threads(threads_per_block, threads_per_block);
	dim3 number_of_blocks(width / threads_per_block, height / threads_per_block);
	size_t buffer_size = width * height * sizeof(Color);
	// Pointers
	Color *buffer;
	ITargetObject **target_objects;
	IObjectList **object_list;
	ILightSource **light_sources;
	ILightSourceEffects **light_source_effects;
	ITargetObject **target_objects_for_light_sources;
	IObjectList **object_list_for_light_sources;

	//Memory allocation
	CUDAMemory<Color>::allocate_managed_instance(buffer, buffer_size);

	CUDAMemory<ITargetObject>::allocate_pointer_to_instance(target_objects, number_of_objects);
	CUDAMemory<IObjectList>::allocate_pointer_to_instance(object_list, 1);

	CUDAMemory<ITargetObject>::allocate_pointer_to_instance(target_objects_for_light_sources, number_of_objects);
	CUDAMemory<IObjectList>::allocate_pointer_to_instance(object_list_for_light_sources, 1);

	CUDAMemory<ILightSource>::allocate_pointer_to_instance(light_sources, number_of_light_sources);
	CUDAMemory<ILightSourceEffects>::allocate_pointer_to_instance(light_source_effects, 1);

	// Object creation on GPU

	create_objects<<<1, 1>>>(target_objects, object_list, number_of_objects);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	create_objects<<<1, 1>>>(target_objects_for_light_sources, object_list_for_light_sources, number_of_objects);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	create_light_sources<<<1, 1>>>(light_sources, light_source_effects, object_list_for_light_sources);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Computing image buffer on GPU
	render_it<<<number_of_blocks, number_of_threads>>>(buffer, width, height, object_list, light_source_effects);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Output the buffer
	std::ofstream ofs;
	ofs.open("./image_cuda.ppm");
	ofs << "P6\n" << width << " " << height << "\n255\n";
	for (size_t pixel_index = 0; pixel_index < width * height; ++pixel_index) {
		for (size_t color_index = 0; color_index < 3; color_index++) {
			ofs << static_cast<char>(255 * std::max(0.f, std::min(1.f, buffer[pixel_index][color_index])));
		}
	}
	// Free GPU memory
	CUDAMemory<Color>::release(buffer);
	checkCudaErrors(cudaGetLastError());
	release_target_objects<<<1, 1>>>(target_objects, number_of_objects);
	checkCudaErrors(cudaGetLastError());
	release_object_list<<<1, 1>>>(object_list, 1);
	checkCudaErrors(cudaGetLastError());
	cudaDeviceReset();
	return 0;
}
