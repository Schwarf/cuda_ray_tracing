//
// Created by andreas on 21.12.21.
//

#ifndef CREATESCENE_CUH
#define CREATESCENE_CUH

#include "./../objects/sphere.cuh"
#include "./../objects/object_list.cuh"
#include "./../materials/material.cuh"
#include "../rays/interfaces/i_light_source.cuh"
#include "../rays/interfaces/i_light_source_effects.cuh"
#include "../rays/light_source.cuh"
#include "../rays/light_source_effects.cuh"


__device__ __host__ inline void build_material(IMaterial * const p_material)
{
	p_material->set_specular_reflection(0.3f);
	p_material->set_diffuse_reflection(0.6f);
	p_material->set_ambient_reflection(0.2f);
	p_material->set_shininess(50.0f);
	p_material->set_transparency(0.0001f);
	p_material->set_refraction_index(1.0f);
	Color red = Vector3D{0.9f, 0.2f, 0.3f};
	p_material->set_rgb_color(red);
}


__device__ __host__ inline void build_material2(IMaterial * const p_material)
{
	p_material->set_specular_reflection(0.3f);
	p_material->set_diffuse_reflection(0.5f);
	p_material->set_ambient_reflection(0.1f);
	p_material->set_shininess(50.0f);
	p_material->set_transparency(0.0001f);
	p_material->set_refraction_index(1.0f);
	Color green = Vector3D{0.3f, 0.9f, 0.3f};
	p_material->set_rgb_color(green);
}

__device__ __host__ inline void build_material3(IMaterial * const p_material)
{
	p_material->set_specular_reflection(0.2f);
	p_material->set_diffuse_reflection(0.6f);
	p_material->set_ambient_reflection(0.1f);
	p_material->set_shininess(50.0f);
	p_material->set_transparency(0.0001f);
	p_material->set_refraction_index(1.0f);
	Color light_blue = Vector3D{0.6f, 0.9f, 0.9f};
	p_material->set_rgb_color(light_blue);
}


__device__ __host__ inline void build_material4(IMaterial * const p_material)
{
	p_material->set_specular_reflection(0.8f);
	p_material->set_diffuse_reflection(0.6f);
	p_material->set_ambient_reflection(0.12f);
	p_material->set_shininess(10.0f);
	p_material->set_transparency(0.0001f);
	p_material->set_refraction_index(1.0f);
	Color whitish = Vector3D{0.9f, 0.7f, 0.7f};
	p_material->set_rgb_color(whitish);
}


__device__ __host__ inline void build_material5(IMaterial * const p_material)
{
	p_material->set_specular_reflection(10.f);
	p_material->set_diffuse_reflection(0.6f);
	p_material->set_ambient_reflection(0.5f);
	p_material->set_shininess(1200.0f);
	p_material->set_transparency(0.0001f);
	p_material->set_refraction_index(1.0f);
	Color whitish = Vector3D{0.3f, 0.3f, 0.3f};
	p_material->set_rgb_color(whitish);
}



__global__ void inline create_objects(ITargetObject **target_objects, IObjectList ** object_list)
{
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		auto sphere_center = Vector3D{-7.f, 5.5f, -20.f};
		auto sphere_radius = 1.5f;
		auto sphere_center2 = Vector3D{0.5f, -1.5f, -15.f};
		auto sphere_radius2 = 4.5f;
		auto sphere_center3 = Vector3D{-1.5f, -2.f, -10.f};
		auto sphere_radius3 = 0.5f;
		auto sphere_center4 = Vector3D{2.f, 1.5f, -9.f};
		auto sphere_radius4 = 1.5f;
		auto sphere_center5 = Vector3D{-5.5f, -4.f, -10.f};
		auto sphere_radius5 = 1.f;


		IMaterial *p_material = new Material();
		IMaterial *p_material2 = new Material();
		IMaterial *p_material3 = new Material();
		IMaterial *p_material4 = new Material();
		IMaterial *p_material5 = new Material();

		build_material(p_material);
		build_material2(p_material2);
		build_material3(p_material3);
		build_material4(p_material4);
		build_material5(p_material5);

		target_objects[0] = new Sphere(sphere_center, sphere_radius, p_material);
		target_objects[1] = new Sphere(sphere_center2, sphere_radius2, p_material2);
		target_objects[2] = new Sphere(sphere_center3, sphere_radius3, p_material3);
		target_objects[3] = new Sphere(sphere_center4, sphere_radius4, p_material4);
		//target_objects[4] = new Sphere(sphere_center5, sphere_radius5, p_material5);

		*object_list = new ObjectList(target_objects, 4);
	}
}

__global__ void inline create_light_sources(ILightSource **light_sources, ILightSourceEffects ** light_source_effects, IObjectList ** object_list)
{
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		auto light_source_position1 = Point3D{-20.f, -20.f, 20.f};
		float light_intensity1 = 1.5;
		ILightSource * light_source1 = new LightSource(light_source_position1, light_intensity1);

		auto light_source_position2 = Point3D{30.f, -50.f, -25.f};
		float light_intensity2 = 1.8;
		ILightSource * light_source2 = new LightSource(light_source_position2, light_intensity2);

		auto light_source_position3 = Point3D{30.f, 20.f, 30.f};
		float light_intensity3 = 1.7;
		ILightSource * light_source3 = new LightSource(light_source_position3, light_intensity3);

		light_sources[0] = light_source1;
		light_sources[1] = light_source2;
		light_sources[2] = light_source3;
		*light_source_effects = new LightSourceEffects(light_sources, object_list, 3);
	}
}



__global__ void release_target_objects(ITargetObject **target_objects, size_t size)
{
	for (int i = 0; i < size; ++i) {
		delete target_objects[i];
	}
	delete *target_objects;
}

__global__ void release_object_list(IObjectList **object_list, size_t size)
{
	for (int i = 0; i < size; ++i) {
		delete object_list[i];
	}
	delete *object_list;
}

#endif