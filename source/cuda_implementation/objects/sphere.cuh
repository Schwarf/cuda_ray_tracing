//
// Created by andreas on 03.10.21.
//

#ifndef SIMPLE_RAY_TRACING_SPHERE_H
#define SIMPLE_RAY_TRACING_SPHERE_H
#include "interfaces/i_sphere.cuh"
#include "./../materials/interfaces/i_material.cuh"

class Sphere final: public ISphere
{
public:
	__device__ Sphere(c_vector3 &center, float radius, IMaterial * material);
	__device__ Sphere(const Sphere &) = default;
	__device__ Sphere(Sphere &&) = default;
	__device__ c_vector3 center() const final;

	__device__ float radius() const final;

	__device__ ~Sphere() override = default;

	__device__ bool does_ray_intersect(const IRay &ray, c_vector3 &hit_normal, c_vector3 &hit_point) const final;

	__device__ IMaterial * material() const final;

private:
	void init() const;
	c_vector3 center_;
	float radius_;
	IMaterial * material_;
};


#endif //SIMPLE_RAY_TRACING_SPHERE_H
