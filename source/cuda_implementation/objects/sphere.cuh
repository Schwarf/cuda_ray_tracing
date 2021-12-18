//
// Created by andreas on 03.10.21.
//

#ifndef CUDA_RAY_TRACING_SPHERE_H
#define CUDA_RAY_TRACING_SPHERE_H
#include "interfaces/i_sphere.cuh"
#include "./../materials/interfaces/i_material.cuh"

class Sphere final: public ISphere
{
public:
	__device__ __host__ Sphere(Vector3D &center, float radius, const IMaterial * const material);
	__device__ __host__ Sphere(const Sphere &) = default;
	__device__ __host__ Sphere(Sphere &&) = default;
	__device__ __host__ Vector3D center() const final;

	__device__ __host__ float radius() const final;

	__device__ __host__ ~Sphere() override = default;

	__device__ __host__ bool does_ray_intersect( const IRay & ray,  IHitRecord & hit_record) const final;

	__device__ __host__ const IMaterial * material() const final;

private:
	Vector3D center_;
	float radius_;
	const IMaterial * material_;
};


#endif //CUDA_RAY_TRACING_SPHERE_H
