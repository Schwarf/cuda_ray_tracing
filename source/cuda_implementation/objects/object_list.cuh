//
// Created by andreas on 18.12.21.
//

#ifndef OBJECT_LIST_CUH
#define OBJECT_LIST_CUH
#include "./interfaces/i_object_list.cuh"

class ObjectList final : public IObjectList
{
public:
	__device__ __host__ ObjectList(ITargetObject** &list, size_t size);
	__device__ __host__ bool hit_by_ray(const IRay &ray, IHitRecord &hit_record) const final;
	__device__ __host__ ~ObjectList() final = default;
private:
	ITargetObject ** object_list_;
	size_t number_of_objects_{};
};


#endif //OBJECT_LIST_CUH
