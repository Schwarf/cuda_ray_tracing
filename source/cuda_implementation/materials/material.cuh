//
// Created by andreas on 02.10.21.
//

#ifndef CUDA_RAY_TRACING_MATERIAL_H
#define CUDA_RAY_TRACING_MATERIAL_H

#include "interfaces/i_material.cuh"
#include "./../miscellaneous/templates/n_tuple.cuh"

class Material final: public IMaterial
{
public:
	__device__ __host__ Material() = default;
	__device__ __host__ Vector3D rgb_color() const final;
	__device__ __host__ float specular_reflection() const final;
	__device__ __host__ float diffuse_reflection() const final;
	__device__ __host__ float ambient_reflection() const final;
	__device__ __host__ float shininess() const final;
	__device__ __host__ float refraction_coefficient() const final;
	__device__ __host__ float specular_exponent() const final;
	__device__ __host__ void set_rgb_color(Vector3D color) final;
	__device__ __host__ void set_specular_reflection(float specular_coefficient) final;
	__device__ __host__ void set_diffuse_reflection(float diffuse_coefficient) final;
	__device__ __host__ void set_ambient_reflection(float ambient_coefficient) final;
	__device__ __host__ void set_refraction_coefficient(float refraction_coefficient) final;
	__device__ __host__ void set_specular_exponent(float specular_exponent) final;
	__device__ __host__ void set_shininess(float shininess) final;
	__device__ __host__ ~Material() override = default;

private:
	float specular_reflection_{-1.0};
	float diffuse_reflection_{-1.0};
	float ambient_reflection_{-1.0};
	float shininess_{-1.0};
	float specular_exponent_{-1.0};
	Vector3D rgb_color_{0., 0., 0.};
	float refraction_coefficient_{-1.0};
};


#endif //CUDA_RAY_TRACING_MATERIAL_H

