//
// Created by andreas on 18.12.21.
//

#ifndef I_OBJECT_LIST_CUH
#define I_OBJECT_LIST_CUH
#include "i_target_object.cuh"

class IObjectList
{
public:
	__device__ __host__ virtual bool hit_by_ray(const IRay &ray, IHitRecord &hit_record) const = 0;
	__device__ __host__ virtual ~IObjectList() = default;
};

#endif //I_OBJECT_LIST_CUH
