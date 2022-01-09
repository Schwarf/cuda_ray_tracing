//
// Created by andreas on 01.11.21.
//

#include "cuda_implementation/miscellaneous/CUDAMemory.cuh"
#include "cuda_implementation/create_scene/create_scene.cuh"
#include "cuda_implementation/camera/camera.cuh"
#include <fstream>
#include <iostream>


__global__ void render_it(Vector3D *buffer, size_t image_width, size_t image_height, IObjectList **object_list,
						  ILightSourceEffects **light_source_effects)
{
	size_t width = threadIdx.x + blockIdx.x * blockDim.x;
	size_t height = threadIdx.y + blockIdx.y * blockDim.y;

	//size_t width = threadIdx.x;
	//size_t height = blockIdx.x;
	if ((width >= image_width) || (height >= image_height)) {
		return;
	}
	float x_direction = float(width) - float(image_width) / 2.f;
	float y_direction = float(height) - float(image_height) / 2.f;
	float z_direction = -float(image_height + image_width) / 2.f;

	Vector3D direction = Vector3D{x_direction, y_direction, z_direction}.normalize();
	Point3D origin = Point3D{0, 0, 0};
	auto ray = Ray(origin, direction);
	auto camera = Camera(image_width, image_height, 2.f, 1.f);
	Color pixel_color = camera.get_pixel_color(ray, object_list, light_source_effects);
	size_t pixel_index = height * image_width + width;
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
