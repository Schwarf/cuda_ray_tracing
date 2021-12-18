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
	__device__ __host__ float refraction_index() const final;
	__device__ __host__ void set_rgb_color(Vector3D color) final;
	__device__ __host__ void set_specular_reflection(float specular_coefficient) final;
	__device__ __host__ void set_diffuse_reflection(float diffuse_coefficient) final;
	__device__ __host__ void set_ambient_reflection(float ambient_coefficient) final;
	__device__ __host__ void set_refraction_index(float refraction_index) final;
	__device__ __host__ void set_shininess(float shininess) final;
	__device__ __host__ ~Material() override = default;
	__device__ __host__ float transparency() const override;
	__device__ __host__ void set_transparency(float transparency) override;

private:
	float transparency_{};
	float diffuse_reflection_{};
	float ambient_reflection_{};
	float shininess_{};
	float specular_reflection_{};
	Vector3D rgb_color_{0., 0., 0.};
	float refraction_index_{};
};


#endif //CUDA_RAY_TRACING_MATERIAL_H

