//
// Created by andreas on 04.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_GEOMETRIC_OBJECT_H
#define SIMPLE_RAY_TRACING_I_GEOMETRIC_OBJECT_H
#include "../../rays/interfaces/i_ray_intersection.cuh"

class ITargetObject: public IRayIntersection
{
public:
	__device__ virtual ~ITargetObject() = default;
};

#endif //SIMPLE_RAY_TRACING_I_GEOMETRIC_OBJECT_H
