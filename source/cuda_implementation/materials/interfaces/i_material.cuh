//
// Created by andreas on 03.10.21.
//

#ifndef CUDA_RAY_TRACING_I_MATERIAL_H
#define CUDA_RAY_TRACING_I_MATERIAL_H
#include "./../../miscellaneous/templates/n_tuple.cuh"
#include "i_reflection_coefficients.cuh"
#include "i_specular_exponent.cuh"
#include "i_refraction_coefficient.cuh"


class IMaterial: public IRefractionCoefficient, public ISpecularExponent, public IReflectionCoefficients
{
public:
	__device__ __host__ virtual Vector3D rgb_color() const = 0;
	__device__ __host__ virtual void set_rgb_color(Vector3D color) = 0;
	__device__ __host__ ~IMaterial() = default;
};
#endif //CUDA_RAY_TRACING_I_MATERIAL_H
