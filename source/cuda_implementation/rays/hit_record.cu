//
// Created by andreas on 18.12.21.
//

#include "hit_record.cuh"
__device__ __host__ Vector3D HitRecord::hit_normal() const
{
	return hit_normal_;
}
__device__ __host__ Point3D HitRecord::hit_point() const
{
	return hit_point_;
}
__device__ __host__ void HitRecord::set_hit_normal(const Vector3D &hit_normal)
{
	hit_normal_ = hit_normal;
}
__device__ __host__ void HitRecord::set_hit_point(const Point3D &hit_point)
{
	hit_point_ = hit_point;
}
__device__ __host__ const IMaterial *HitRecord::get_material() const
{
	return material_;
}
__device__ __host__ void HitRecord::set_material(const IMaterial *material)
{
	material_ = material;
}
