//
// Created by andreas on 03.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_REFRACTION_COEFFICIENT_H
#define SIMPLE_RAY_TRACING_I_REFRACTION_COEFFICIENT_H
class IRefractionCoefficient
{
public:
	__device__ virtual float refraction_coefficient() const = 0;
	__device__ virtual void set_refraction_coefficient(float refraction_coefficient) = 0;
	__device__ virtual ~IRefractionCoefficient() = default;
};
#endif //SIMPLE_RAY_TRACING_I_REFRACTION_COEFFICIENT_H
