//
// Created by andreas on 21.12.21.
//

#include "object_list.cuh"
__device__ __host__ ObjectList::ObjectList(ITargetObject **&list, size_t size)
	: object_list_(list), number_of_objects_(size)
{

}
__device__ __host__ bool ObjectList::hit_by_ray(const IRay &ray, IHitRecord &hit_record) const
{
	bool is_hit{};
	for(size_t index = 0; index < number_of_objects_; ++index)
	{
		if(object_list_[index]->does_ray_intersect(ray, hit_record))
			is_hit =true;
	}
	return is_hit;
}

