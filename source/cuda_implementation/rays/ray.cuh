//
// Created by andreas on 03.10.21.
//

#ifndef CUDA_RAY_TRACING_RAY_H
#define CUDA_RAY_TRACING_RAY_H

#include "interfaces/i_ray.cuh"
#include "../miscellaneous/templates/n_tuple.cuh"

class Ray final: public IRay
{
public:
	__device__ __host__ Ray(Point3D &origin, Vector3D &direction);
	__device__ __host__ Vector3D direction_normalized() const final;
	__device__ __host__ Point3D origin() const final;
 	__device__ __host__ ~Ray() final = default;
private:
	Vector3D direction_normalized_;
	Point3D origin_;
};


#endif //CUDA_RAY_TRACING_RAY_H
