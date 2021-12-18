//
// Created by andreas on 02.10.21.
//

#include "material.cuh"


__device__ __host__ Vector3D Material::rgb_color() const
{
	return rgb_color_;
}
__device__ __host__ float Material::specular_reflection() const
{
	return specular_exponent_;
}
__device__ __host__ float Material::diffuse_reflection() const
{
	return diffuse_reflection_;
}
__device__ __host__ float Material::ambient_reflection() const
{
	return ambient_reflection_;
}
__device__ __host__ float Material::shininess() const
{
	return shininess_;
}
__device__ __host__ void Material::set_specular_reflection(float specular_coefficient)
{
	specular_reflection_ = specular_coefficient;
}
__device__ __host__ void Material::set_diffuse_reflection(float diffuse_coefficient)
{
	diffuse_reflection_ = diffuse_coefficient;
}
__device__ __host__ void Material::set_ambient_reflection(float ambient_coefficient)
{
	ambient_reflection_ = ambient_coefficient;
}
__device__ __host__ void Material::set_shininess(float shininess)
{
	 shininess_ = shininess;
}
__device__ __host__ float Material::refraction_coefficient() const
{
	return refraction_coefficient_;
}
__device__ __host__ void Material::set_refraction_coefficient(float refraction_coefficient)
{
	refraction_coefficient_ = refraction_coefficient;
}
__device__ __host__ float Material::specular_exponent() const
{
	return specular_reflection_;
}
__device__ __host__ void Material::set_specular_exponent(float specular_exponent)
{
	specular_exponent_ = specular_exponent;
}
__device__ __host__ void Material::set_rgb_color(Vector3D color)
{
	rgb_color_ = color;
}
