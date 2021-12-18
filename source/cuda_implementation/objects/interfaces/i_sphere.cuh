//
// Created by andreas on 03.10.21.
//

#ifndef CUDA_RAY_TRACING_I_SPHERE_H
#define CUDA_RAY_TRACING_I_SPHERE_H
#include "./../../miscellaneous/templates/n_tuple.cuh"
#include "i_target_object.cuh"
#include "../../materials/interfaces/i_material.cuh"

class ISphere: public ITargetObject
{
public:
	__device__ __host__ virtual Vector3D center() const = 0;
	__device__ __host__ virtual float radius() const = 0;
	__device__ __host__ virtual const IMaterial * material() const = 0;
	__device__ __host__ virtual ~ISphere() = default;
};
#endif //CUDA_RAY_TRACING_I_SPHERE_H
