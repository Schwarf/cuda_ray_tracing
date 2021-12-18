//
// Created by andreas on 03.10.21.
//

#ifndef CUDA_RAY_TRACING_I_MATERIAL_H
#define CUDA_RAY_TRACING_I_MATERIAL_H
#include "./../../miscellaneous/templates/n_tuple.cuh"
#include "i_phong_reflection_coefficients.cuh"
#include "i_refraction_coefficients.cuh"


class IMaterial: public IRefractionCoefficient, public IPhongReflectionCoefficients
{
public:
	__device__ __host__ virtual Vector3D rgb_color() const = 0;
	__device__ __host__ virtual void set_rgb_color(Vector3D color) = 0;
	__device__ __host__ ~IMaterial() = default;
};
#endif //CUDA_RAY_TRACING_I_MATERIAL_H
