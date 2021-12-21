//
// Created by andreas on 01.11.21.
//

#include "cuda_implementation/miscellaneous/templates/n_tuple.cuh"
#include "cuda_implementation/rays/ray.cuh"
#include "cuda_implementation/rays/hit_record.cuh"
#include "cuda_implementation/rays/ray_interactions.cuh"
#include "cuda_implementation/miscellaneous/CUDAMemory.cuh"
#include "cuda_implementation/objects/interfaces/i_object_list.cuh"
#include "cuda_implementation/create_scene/create_scene.cuh"
#include <fstream>
#include <iostream>

#define checkCudaErrors(value) check_cuda( (value), #value, __FILE__, __LINE__)

__device__ __host__ Color get_pixel_color(const IRay &ray,
										  IObjectList ** const object_list,
										  IHitRecord &hit_record,
										  IRayInteractions & ray_interaction,
										  size_t recursion_depth)
{
	auto is_hit = (*object_list)->any_object_hit_by_ray(ray, hit_record);
	if (!is_hit) {
		return Color{0.2, 0.7, 0.8};
	}
	recursion_depth--;
	auto diffuse_ray = Ray();
	auto specular_ray = Ray();
	ray_interaction.diffuse_scatter(hit_record, diffuse_ray);
	ray_interaction.specular_scatter(ray, hit_record, specular_ray);
//	auto reflected_color = get_pixel_color(diffuse_ray, sphere, hit_record, ray_interaction, recursion_depth);
//	auto refracted_color = get_pixel_color(specular_ray, sphere, hit_record, ray_interaction, recursion_depth);
	Color diffuse_color = hit_record.get_material()->diffuse_reflection() * hit_record.get_material()->rgb_color();
	Color white = Color{1, 1, 1};
	Color specular_color = white * hit_record.get_material()->specular_reflection();
	return diffuse_color + specular_color;
}

__global__ void render_it(Vector3D *buffer, size_t max_width, size_t max_height, IObjectList ** object_list)
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
	auto ray_interactions = RayInteractions();
	Color pixel_color = get_pixel_color(ray, object_list, hit_record, ray_interactions, 2);
	size_t pixel_index = height * max_width + width;
	buffer[pixel_index] = pixel_color;

}

int main()
{
	size_t width = 1024;
	size_t height = 768;
	// Why is 32 the maximum number of threads per block
	//constexpr size_t threads_per_block = 32;
	//dim3 number_of_threads(threads_per_block, threads_per_block);
	//dim3 number_of_blocks(width / threads_per_block, height / threads_per_block);
	size_t number_of_blocks = 768;
	size_t number_of_threads{1024};
	size_t buffer_size = width * height * sizeof(Color);
	std::cout << buffer_size << std::endl;
	Color *buffer;
	CUDAMemory<Color>::allocate_managed_instance(buffer, buffer_size);
	ITargetObject ** target_objects;
	IObjectList **object_list;
	CUDAMemory<IObjectList>::allocate_pointer_to_instance(object_list, 1);
	CUDAMemory<ITargetObject>::allocate_pointer_to_instance(target_objects, 3);
	//cudaMallocManaged((void **)&buffer, buffer_size);
	create_objects<<<1,1>>>(target_objects, object_list);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors( cudaDeviceSynchronize());

	render_it<<<number_of_blocks, number_of_threads>>>(buffer, width, height, object_list);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors( cudaDeviceSynchronize());
	std::ofstream ofs;
	ofs.open("./cuda_image.ppm");
	ofs << "P6\n" << width << " " << height << "\n255\n";
	for (size_t pixel_index = 0; pixel_index < width * height; ++pixel_index) {
		for (size_t color_index = 0; color_index < 3; color_index++) {
			ofs << static_cast<char>(255 * std::max(0.f, std::min(1.f, buffer[pixel_index][color_index])));
		}
	}
	CUDAMemory<Color>::release(buffer);
	//CUDAMemory<ITargetObject>::release_pointer_to_instance(target_objects);
	return 0;
}