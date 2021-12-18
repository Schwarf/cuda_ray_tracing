//
// Created by andreas on 18.12.21.
//

#ifndef LIGHT_SOURCE_CUH
#define LIGHT_SOURCE_CUH
#include "interfaces/i_light_source.cuh"
#include "../miscellaneous/templates/n_tuple.cuh"

class LightSource final: public ILightSource
{
public:
	__device__ __host__ LightSource(const Point3D &position, float intensity);
	__device__ __host__ Point3D position() const final;
	__device__ __host__ float intensity() const final;
	__device__ __host__ ~LightSource() override = default;

private:
	Point3D position_;
	float intensity_;
};


#endif //LIGHT_SOURCE_CUH
