//
// Created by andreas on 02.10.21.
//

#ifndef CUDA_RAY_TRACING_I_REFLECTION_COEFFICIENTS_H
#define CUDA_RAY_TRACING_I_REFLECTION_COEFFICIENTS_H


class IReflectionCoefficients
{
public:
	__device__ __host__ virtual float specular_reflection() const = 0;

	__device__ __host__ virtual float diffuse_reflection() const = 0;

	__device__ __host__ virtual float ambient_reflection() const = 0;

	__device__ __host__ virtual float shininess() const = 0;

	__device__ __host__ virtual void set_specular_reflection(float specular_coefficient) = 0;

	__device__ __host__ virtual void set_diffuse_reflection(float diffuse_coefficient) = 0;

	__device__ __host__ virtual void set_ambient_reflection(float ambient_coefficient)  = 0;

	__device__ __host__ virtual void set_shininess(float shininess) = 0;

	__device__ __host__ virtual ~IReflectionCoefficients() = default;
};

#endif //CUDA_RAY_TRACING_I_REFLECTION_COEFFICIENTS_H
