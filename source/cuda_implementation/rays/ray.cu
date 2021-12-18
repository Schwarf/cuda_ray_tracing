//
// Created by andreas on 03.10.21.
//

#include "ray.cuh"

__device__ __host__ Ray::Ray(Point3D &origin, Vector3D &direction)
{
	direction_normalized_ = direction.normalize();
	origin_ = origin;
}

__device__ __host__ Vector3D Ray::direction_normalized() const
{
	return direction_normalized_;
}

__device__ __host__ Point3D Ray::origin() const
{
	return origin_;
}

