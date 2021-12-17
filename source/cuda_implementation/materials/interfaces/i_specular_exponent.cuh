//
// Created by andreas on 03.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_SPECULAR_EXPONENT_H
#define SIMPLE_RAY_TRACING_I_SPECULAR_EXPONENT_H
class ISpecularExponent
{
public:
	__device__ virtual float specular_exponent() const = 0;
	__device__ virtual void set_specular_exponent(float specular_exponent) = 0;
	__device__ virtual ~ISpecularExponent() = default;
};
#endif //SIMPLE_RAY_TRACING_I_SPECULAR_EXPONENT_H
