//
// Created by andreas on 03.10.21.
//

#ifndef CUDA_RAY_TRACING_I_SPECULAR_EXPONENT_H
#define CUDA_RAY_TRACING_I_SPECULAR_EXPONENT_H
class ISpecularExponent
{
public:
	__device__ __host__ virtual float specular_exponent() const = 0;
	__device__ __host__ virtual void set_specular_exponent(float specular_exponent) = 0;
	__device__ __host__ virtual ~ISpecularExponent() = default;
};
#endif //CUDA_RAY_TRACING_I_SPECULAR_EXPONENT_H
