//
// Created by andreas on 18.12.21.
//

#ifndef HIT_RECORD_CUH
#define HIT_RECORD_CUH

#include "interfaces/i_hit_record.cuh"
#include "../miscellaneous/templates/n_tuple.cuh"

class HitRecord final : public IHitRecord
{
public:
	__device__ __host__ HitRecord() = default;
	__device__ __host__ Vector3D hit_normal() const final;
	__device__ __host__ Point3D hit_point() const final;
	__device__ __host__ void set_hit_normal(const Vector3D &hit_normal) final;
	__device__ __host__ void set_hit_point(const Point3D &hit_point) final;
	__device__ __host__ ~HitRecord() final =default;
	__device__ __host__ const IMaterial *get_material() const final;
	__device__ __host__ void set_material(const IMaterial *material) final;

private:
	Vector3D hit_normal_{0,0,0};
	Point3D hit_point_{0,0,0};
	const IMaterial * material_ = nullptr;
};


#endif //HIT_RECORD_CUH
