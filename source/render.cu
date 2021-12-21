//
// Created by andreas on 01.11.21.
//

#include "cuda_implementation/miscellaneous/templates/n_tuple.cuh"
#include "cuda_implementation/rays/ray.cuh"
#include "cuda_implementation/rays/hit_record.cuh"
#include "cuda_implementation/objects/sphere.cuh"
#include "cuda_implementation/objects/object_list.cuh"
#include <fstream>
#include "cuda_implementation/materials/material.cuh"
#include "cuda_implementation/rays/ray_interactions.cuh"
#include <iostream>
#include "cuda_implementation/miscellaneous/CUDAMemory.cuh"

#define checkCudaErrors(value) check_cuda( (value), #value, __FILE__, __LINE__)

__device__ __host__ void build_material(IMaterial * const p_material)
{
	p_material->set_specular_reflection(0.3f);
	p_material->set_diffuse_reflection(0.6);
	p_material->set_ambient_reflection(0.3);
	p_material->set_shininess(50.0);
	p_material->set_transparency(0.0001);
	p_material->set_refraction_index(1.0);
	Vector3D color = Vector3D{0.9, 0.2, 0.3};
	p_material->set_rgb_color(color);
}


__device__ __host__ void build_material2(IMaterial * const p_material)
{
	p_material->set_specular_reflection(0.5f);
	p_material->set_diffuse_reflection(0.2);
	p_material->set_ambient_reflection(0.1);
	p_material->set_shininess(50.0);
	p_material->set_transparency(0.0001);
	p_material->set_refraction_index(1.0);
	Vector3D color = Vector3D{0.3, 0.9, 0.3};
	p_material->set_rgb_color(color);
}


__global__ void create_objects(ITargetObject ** target_objects, ObjectList *object_list)
{
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		auto sphere_center = Vector3D{-3.5f, 3.5f, -15.f};
		auto sphere_radius = 1.5f;
		auto sphere_center2 = Vector3D{0.5f, -1.5f, -10.f};
		auto sphere_radius2 = 2.5f;

		Material material;
		IMaterial *p_material = &material;

		Material material2;
		IMaterial *p_material2 = &material2;

		build_material(p_material);
		build_material2(p_material2);
		auto sphere = Sphere(sphere_center, sphere_radius, p_material);
		auto sphere2 = Sphere(sphere_center2, sphere_radius2, p_material2);

		auto p_sphere = &sphere;
		auto p_sphere2 = &sphere2;
		target_objects[0] = p_sphere;
		target_objects[1] = p_sphere2;
		object_list = new ObjectList(target_objects, 2);
	}
}

__device__ __host__ Color get_pixel_color(const IRay &ray,
										  ObjectList &object_list,
										  IHitRecord &hit_record,
										  IRayInteractions & ray_interaction,
										  size_t recursion_depth)
{
	auto is_hit = object_list.hit_by_ray(ray, hit_record);
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

__global__ void render_it(Vector3D *buffer, size_t max_width, size_t max_height, ITargetObject ** object_list)
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
	auto sphere_center = Vector3D{-3.5f, 3.5f, -15.f};
	auto sphere_radius = 1.5f;
	auto sphere_center2 = Vector3D{0.5f, -1.5f, -10.f};
	auto sphere_radius2 = 2.5f;

	Material material;
	IMaterial * p_material = & material;

	Material material2;
	IMaterial * p_material2 = & material2;

	build_material(p_material);
	build_material2(p_material2);
	auto sphere = Sphere(sphere_center, sphere_radius, p_material);
	auto sphere2 = Sphere(sphere_center2, sphere_radius2, p_material2);


	auto p_sphere = & sphere;
	auto p_sphere2 = & sphere2;
	object_list[0] = p_sphere;
	object_list[1] = p_sphere2;
	auto list = ObjectList(object_list, 2);

	Vector3D direction = Vector3D{x_direction, y_direction, z_direction}.normalize();
	Vector3D origin = Vector3D{0, 0, 0};
	auto ray = Ray(origin, direction);
	auto hit_record = HitRecord();
	auto ray_interactions = RayInteractions();
	Color pixel_color = get_pixel_color(ray, list, hit_record, ray_interactions, 2);
	size_t pixel_index = height * max_width + width;
	buffer[pixel_index] = pixel_color;

}

int main()
{
	size_t width = 1024;
	size_t height = 768;
	// Why is 32 the maximum number of threads per block
	constexpr size_t threads_per_block = 32;
	//dim3 number_of_threads(threads_per_block, threads_per_block);

	//dim3 number_of_blocks(width / threads_per_block, height / threads_per_block);
	int number_of_blocks = 768;
	int number_of_threads{1024};
	size_t buffer_size = width * height * sizeof(float3);
	std::cout << buffer_size << std::endl;
	Color *buffer;
	CUDAMemory<Color>::allocate_managed_instance(buffer, buffer_size);
	ITargetObject ** target_objects;
	//ObjectList *object_list;
	//CUDAMemory<ObjectList>::allocate_instance(object_list, 1);
	CUDAMemory<ITargetObject>::allocate_pointer_to_instance(target_objects, 2);
	//cudaMallocManaged((void **)&buffer, buffer_size);
	//create_objects<<<1,1>>>(target_objects, object_list);

	render_it<<<number_of_blocks, number_of_threads>>>(buffer, width, height, target_objects);
	cudaGetLastError();
	cudaDeviceSynchronize();
	std::ofstream ofs;
	ofs.open("./cuda_image.ppm");
	ofs << "P6\n" << width << " " << height << "\n255\n";
	for (size_t pixel_index = 0; pixel_index < width * height; ++pixel_index) {
		for (size_t color_index = 0; color_index < 3; color_index++) {
			ofs << static_cast<char>(255 * std::max(0.f, std::min(1.f, buffer[pixel_index][color_index])));
		}
	}CUDAMemory<Color>::release(buffer);

	return 0;
}