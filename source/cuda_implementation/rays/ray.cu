//
// Created by andreas on 03.10.21.
//

#include "ray.cuh"

__device__ __host__ Ray::Ray(const Point3D &origin, const Vector3D &direction)
{
	direction_ = direction;
	origin_ = origin;
}

__device__ __host__ Vector3D Ray::direction_normalized() const
{
	auto direction = direction_;
	return direction.normalize();
}

__device__ __host__ Point3D Ray::origin() const
{
	return origin_;
}
__device__ __host__ void Ray::set_direction(const Vector3D &direction)
{
	direction_ = direction;
}
__device__ __host__ void Ray::set_origin(const Point3D &origin)
{
	origin_ = origin;
}
