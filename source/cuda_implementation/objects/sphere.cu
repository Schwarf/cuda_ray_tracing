//
// Created by andreas on 03.10.21.
//

#include "sphere.cuh"

__device__ __host__ Sphere::Sphere(Vector3D &center, float radius, const IMaterial * material)
{
	center_ = center;
	radius_ = radius;
	material_ = material;
}

__device__ __host__ Vector3D Sphere::center() const
{
	return center_;
}

__device__ __host__ float Sphere::radius() const
{
	return radius_;
}


__device__ __host__ bool Sphere::does_ray_intersect(const IRay & ray,  IHitRecord & hit_record) const
{
	Vector3D origin_to_center = (center_ - ray.origin());
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
	hit_record.set_hit_point(ray.origin() + ray.direction_normalized() * closest_hit_distance);
	hit_record.set_hit_normal((hit_record.hit_point() - center_).normalize());
	hit_record.set_material(material_);
	return true;
}

__device__ __host__ const IMaterial * Sphere::material() const
{
	return material_;
}