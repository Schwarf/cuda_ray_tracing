//
// Created by andreas on 18.12.21.
//

#include "light_source.cuh"
__device__ __host__ LightSource::LightSource(const Point3D &position, float intensity)
{
	position_ = position;
	intensity_ = intensity;
}
__device__ __host__ Point3D LightSource::position() const
{
	return position_;
}
__device__ __host__ float LightSource::intensity() const
{
	return intensity_;
}
