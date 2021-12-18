//
// Created by andreas on 03.10.21.
//

#ifndef CUDA_RAY_TRACING_I_REFRACTION_COEFFICIENT_H
#define CUDA_RAY_TRACING_I_REFRACTION_COEFFICIENT_H
class IRefractionCoefficient
{
public:
	__device__ __host__ virtual float refraction_coefficient() const = 0;
	__device__ __host__ virtual void set_refraction_coefficient(float refraction_coefficient) = 0;
	__device__ __host__ virtual ~IRefractionCoefficient() = default;
};
#endif //CUDA_RAY_TRACING_I_REFRACTION_COEFFICIENT_H
