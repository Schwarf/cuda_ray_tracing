//
// Created by andreas on 18.12.21.
//

#ifndef I_HIT_RECORD_CUH
#define I_HIT_RECORD_CUH
#include "./../../miscellaneous/templates/n_tuple.cuh"

class IHitRecord
{
public:
	__device__ __host__ virtual Vector3D hit_normal() const = 0;
	__device__ __host__ virtual Point3D hit_point() const = 0;
	__device__ __host__ virtual void set_hit_normal(const Vector3D & hit_normal)  = 0;
	__device__ __host__ virtual void set_hit_point(const Point3D & hit_point)  = 0;

	__device__ __host__ virtual ~IHitRecord() = default;
};


#endif //I_HIT_RECORD_CUH
