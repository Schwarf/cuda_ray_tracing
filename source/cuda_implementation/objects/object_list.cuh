//
// Created by andreas on 18.12.21.
//

#ifndef OBJECT_LIST_CUH
#define OBJECT_LIST_CUH
#include "./interfaces/i_object_list.cuh"

template<size_t maximal_number_of_objects>
class ObjectList final : public IObjectList
{
public:
	__device__ __host__ void add_object(const ITargetObject * target_object) final{
		if(number_of_objects_ < maximal_number_of_objects) {
			object_list_[number_of_objects_] = target_object;
			number_of_objects_++;
		}
	}

	__device__ __host__ const ITargetObject * object(size_t index) const final
	{
		if (index < number_of_objects_)
			return object_list_[index];
	}
	__device__ __host__ const ITargetObject * get_object_hit_by_ray(const IRay &ray, IHitRecord &hit_record) const final
	{
		for(size_t index = 0; index < number_of_objects_; ++index)
		{
			if(object_list_[index]->does_ray_intersect(ray, hit_record))
				return object_list_[index];
		}
		return nullptr;
	}
	__device__ __host__ size_t number_of_objects() final{
		return number_of_objects_;
	};
	__device__ __host__ ~ObjectList() override = default;
private:
	const ITargetObject * object_list_[maximal_number_of_objects]{};
	size_t number_of_objects_{};
};


#endif //OBJECT_LIST_CUH
