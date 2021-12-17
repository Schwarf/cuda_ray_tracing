//
// Created by andreas on 03.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_SPHERE_H
#define SIMPLE_RAY_TRACING_I_SPHERE_H
#include "./../../miscellaneous/templates/c_vector.cuh"
#include "i_target_object.cuh"
#include "../../materials/interfaces/i_material.cuh"

class ISphere: public ITargetObject
{
public:
	__device__ virtual c_vector3 center() const = 0;
	__device__ virtual float radius() const = 0;
	__device__ virtual IMaterial * material() const = 0;
	__device__ virtual ~ISphere() = default;
};
#endif //SIMPLE_RAY_TRACING_I_SPHERE_H
