//
// Created by andreas on 03.10.21.
//

#ifndef CUDA_RAY_TRACING_I_RAY_H
#define CUDA_RAY_TRACING_I_RAY_H
#include "./../../miscellaneous/templates/n_tuple.cuh"

class IRay
{
public:
	__device__ __host__ virtual Vector3D direction_normalized() const = 0;
	__device__ __host__ virtual Point3D origin() const = 0;
	__device__ __host__ virtual ~IRay() = default;
};
#endif //CUDA_RAY_TRACING_I_RAY_H
