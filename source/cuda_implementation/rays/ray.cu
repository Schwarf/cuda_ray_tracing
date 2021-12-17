//
// Created by andreas on 03.10.21.
//

#include "ray.cuh"

__device__ Ray::Ray(c_vector3 &origin, c_vector3 &direction)
{
	direction_normalized_ = direction.normalize();
	origin_ = origin;
}

__device__ c_vector3 Ray::direction_normalized() const
{
	return direction_normalized_;
}

__device__ c_vector3 Ray::origin() const
{
	return origin_;
}

