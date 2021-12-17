//
// Created by andreas on 03.10.21.
//

#include "sphere.cuh"

__device__ Sphere::Sphere(c_vector3 &center, float radius, IMaterial * material)
{
	center_ = center;
	radius_ = radius;
	material_ = material;
}

__device__ c_vector3 Sphere::center() const
{
	return center_;
}

__device__ float Sphere::radius() const
{
	return radius_;
}


__device__ bool Sphere::does_ray_intersect(const IRay &ray, c_vector3 &hit_normal, c_vector3 &hit_point) const
{
	c_vector3 origin_to_center = (center_ - ray.origin());
	float origin_to_center_dot_direction = origin_to_center * ray.direction_normalized();
	float epsilon = 1e-3;
	float discriminant = origin_to_center_dot_direction * origin_to_center_dot_direction -
		((origin_to_center * origin_to_center) - radius_ * radius_);
	if (discriminant < 0.0) {
		return false;
	}

	float closest_hit_distance = origin_to_center_dot_direction - std::sqrt(discriminant);
	float hit_distance = origin_to_center_dot_direction + std::sqrt(discriminant);
	if (closest_hit_distance < epsilon) {
		closest_hit_distance = hit_distance;
	}
	if (closest_hit_distance < epsilon) {
		return false;
	}
	hit_point = ray.origin() + ray.direction_normalized() * closest_hit_distance;
	hit_normal = (hit_point - center_).normalize();
	return true;
}

__device__ IMaterial * Sphere::material() const
{
	return material_;
}