//
// Created by andreas on 03.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_MATERIAL_H
#define SIMPLE_RAY_TRACING_I_MATERIAL_H
#include "./../../miscellaneous/templates/c_vector.cuh"
#include "i_reflection_coefficients.cuh"
#include "i_specular_exponent.cuh"
#include "i_refraction_coefficient.cuh"


class IMaterial: public IRefractionCoefficient, public ISpecularExponent, public IReflectionCoefficients
{
public:
	__device__ virtual c_vector3 rgb_color() const = 0;
	__device__ virtual void set_rgb_color(c_vector3 color) = 0;
	__device__ ~IMaterial() = default;
};
#endif //SIMPLE_RAY_TRACING_I_MATERIAL_H
