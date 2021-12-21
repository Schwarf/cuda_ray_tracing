//
// Created by andreas on 21.12.21.
//

#ifndef CREATESCENE_CUH
#define CREATESCENE_CUH

#include "./../objects/sphere.cuh"
#include "./../objects/object_list.cuh"
#include "./../materials/material.cuh"


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

__device__ __host__ void build_material3(IMaterial * const p_material)
{
	p_material->set_specular_reflection(0.2f);
	p_material->set_diffuse_reflection(0.3);
	p_material->set_ambient_reflection(0.5);
	p_material->set_shininess(50.0);
	p_material->set_transparency(0.0001);
	p_material->set_refraction_index(1.0);
	Vector3D color = Vector3D{0.6, 0.9, 0.9};
	p_material->set_rgb_color(color);
}



__global__ void create_objects(ITargetObject **target_objects, IObjectList ** object_list)
{
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		auto sphere_center = Vector3D{-3.5f, 3.5f, -15.f};
		auto sphere_radius = 1.5f;
		auto sphere_center2 = Vector3D{0.5f, -1.5f, -10.f};
		auto sphere_radius2 = 2.5f;
		auto sphere_center3 = Vector3D{2.5f, 1.5f, -10.f};
		auto sphere_radius3 = 0.5f;

		IMaterial *p_material = new Material();
		IMaterial *p_material2 = new Material();
		IMaterial *p_material3 = new Material();

		build_material(p_material);
		build_material2(p_material2);
		build_material3(p_material3);

		target_objects[0] = new Sphere(sphere_center, sphere_radius, p_material);
		target_objects[1] = new Sphere(sphere_center2, sphere_radius2, p_material2);
		target_objects[2] = new Sphere(sphere_center3, sphere_radius3, p_material3);
		*object_list = new ObjectList(target_objects, 3);
	}
}
#endif