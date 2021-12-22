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
#include <fstream>
#include <iostream>

__device__ __host__ Color get_pixel_color(const IRay &ray,
										  IObjectList **const object_list,
										  ILightSourceEffects **light_source_effects,
										  IHitRecord &hit_record,
										  size_t recursion_depth)
{
	auto is_hit = (*object_list)->any_object_hit_by_ray(ray, hit_record);
	if (!is_hit) {
		return Color{0.2, 0.7, 0.8};
	}
	float diffuse_intensity{};
	float specular_intensity{};
	(*light_source_effects)
		->compute_light_source_effects(ray, hit_record, diffuse_intensity, specular_intensity);
	Color diffuse_color = diffuse_intensity*hit_record.get_material()->diffuse_reflection() * hit_record.get_material()->rgb_color();
	Color white = Color{1, 1, 1};
	Color specular_color = specular_intensity* white * hit_record.get_material()->specular_reflection();
	return diffuse_color + specular_color;
}

__global__ void render_it(Vector3D *buffer, size_t max_width, size_t max_height, IObjectList **object_list,
						  ILightSourceEffects **light_source_effects)
{
	//size_t width = threadIdx.x + blockIdx.x * blockDim.x;
	//size_t height = threadIdx.y + blockIdx.y * blockDim.y;

	size_t width = threadIdx.x;
	size_t height = blockIdx.x;
	if ((width >= max_width) || (height >= max_height)) {
		return;
	}
	float x_direction = float(width) - float(max_width) / 2.f;
	float y_direction = float(height) - float(max_height) / 2.f;
	float z_direction = -float(max_height + max_width) / 2.f;

	Vector3D direction = Vector3D{x_direction, y_direction, z_direction}.normalize();
	Point3D origin = Point3D{0, 0, 0};
	auto ray = Ray(origin, direction);
	auto hit_record = HitRecord();
	Color pixel_color = get_pixel_color(ray, object_list, light_source_effects, hit_record, 2);
	size_t pixel_index = height * max_width + width;
	buffer[pixel_index] = pixel_color;

}

int main()
{
	constexpr size_t width = 1024;
	constexpr size_t height = 768;
	constexpr size_t number_of_blocks = 768;
	constexpr size_t number_of_threads = 1024;
	constexpr size_t number_of_objects = 4;
	constexpr size_t number_of_light_sources = 3;



	// Why is 32 the maximum number of threads per block
	//constexpr size_t threads_per_block = 32;
	//dim3 number_of_threads(threads_per_block, threads_per_block);
	//dim3 number_of_blocks(width / threads_per_block, height / threads_per_block);
	size_t buffer_size = width * height * sizeof(Color);
	// Pointers
	Color *buffer;
	ITargetObject **target_objects;
	IObjectList **object_list;
	ILightSource **light_sources;
	ILightSourceEffects **light_source_effects;

	//Memory allocation
	CUDAMemory<Color>::allocate_managed_instance(buffer, buffer_size);
	CUDAMemory<IObjectList>::allocate_pointer_to_instance(object_list, 1);
	CUDAMemory<ITargetObject>::allocate_pointer_to_instance(target_objects, number_of_objects);
	CUDAMemory<ILightSourceEffects>::allocate_pointer_to_instance(light_source_effects, 1);
	CUDAMemory<ILightSource>::allocate_pointer_to_instance(light_sources, number_of_light_sources);

	// Object creation on GPU
	create_objects<<<1, 1>>>(target_objects, object_list);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	create_light_sources<<<1, 1>>>(light_sources, light_source_effects);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	// Computing image buffer on GPU
	render_it<<<number_of_blocks, number_of_threads>>>(buffer, width, height, object_list, light_source_effects);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	// Output the buffer
	std::ofstream ofs;
	ofs.open("./cuda_image.ppm");
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
//	CUDAMemory<ITargetObject>::release_pointer_to_instance(target_objects);
//	CUDAMemory<IObjectList>::release_pointer_to_instance(object_list);
	cudaDeviceReset();
return 0;
}