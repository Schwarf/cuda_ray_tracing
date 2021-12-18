	//
// Created by andreas on 04.10.21.
//

#ifndef CUDA_RAY_TRACING_I_GEOMETRIC_OBJECT_H
#define CUDA_RAY_TRACING_I_GEOMETRIC_OBJECT_H
#include "../../rays/interfaces/i_ray_intersection.cuh"

class ITargetObject: public IRayIntersection
{
public:
	__device__ __host__ virtual ~ITargetObject() = default;
};

#endif //CUDA_RAY_TRACING_I_GEOMETRIC_OBJECT_H
