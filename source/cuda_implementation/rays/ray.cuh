//
// Created by andreas on 03.10.21.
//

#ifndef SIMPLE_RAY_TRACING_RAY_H
#define SIMPLE_RAY_TRACING_RAY_H

#include "interfaces/i_ray.cuh"
#include "../miscellaneous/templates/c_vector.cuh"

class Ray final: public IRay
{
public:
	__device__ Ray(c_vector3 &origin, c_vector3 &direction);
	__device__ c_vector3 direction_normalized() const final;

	__device__ c_vector3 origin() const final;

	__device__ ~Ray() final = default;
private:
	c_vector3 direction_normalized_;
	c_vector3 origin_;
};


#endif //SIMPLE_RAY_TRACING_RAY_H
