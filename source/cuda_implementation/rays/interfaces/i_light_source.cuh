//
// Created by andreas on 18.12.21.
//

#ifndef I_LIGHT_SOURCE_CUH
#define I_LIGHT_SOURCE_CUH
#include "./../../miscellaneous/templates/n_tuple.cuh"

class ILightSource
{
public:
	__device__ __host__ virtual Point3D position() const = 0;
	__device__ __host__ virtual float intensity() const = 0;
	__device__ __host__ virtual ~ILightSource() = default;
};

#endif //I_LIGHT_SOURCE_CUH
