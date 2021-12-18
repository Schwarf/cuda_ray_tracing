//
// Created by andreas on 18.12.21.
//

#ifndef I_OBJECT_LIST_CUH
#define I_OBJECT_LIST_CUH
#include "i_target_object.cuh"

class IObjectList
{
public:
	__device__ __host__ virtual void add_object(const ITargetObject * target_object) = 0;
	__device__ __host__ virtual const ITargetObject * object(size_t index) const = 0;
	__device__ __host__ virtual const ITargetObject * get_object_hit_by_ray(const IRay &ray, IHitRecord &hit_record) const = 0;
	__device__ __host__ virtual size_t number_of_objects() = 0;
	__device__ __host__ virtual ~IObjectList() = default;
};

#endif //I_OBJECT_LIST_CUH
