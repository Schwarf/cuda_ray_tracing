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
	__device__ __host__ Ray() = default;
	__device__ __host__ Ray(const Point3D &origin, const Vector3D &direction);
	__device__ __host__ Vector3D direction_normalized() const final;
	__device__ __host__ Point3D origin() const final;
 	__device__ __host__ ~Ray() final = default;
	__device__ __host__ void set_direction(const Vector3D &direction) override;
	__device__ __host__ void set_origin(const Point3D &origin) override;
private:
	Vector3D direction_{0,0,0};
	Point3D origin_{0,0,0};
};


#endif //CUDA_RAY_TRACING_RAY_H
