
// Simpl
// Created by andreas on 02.10.21.
//

#ifndef SOURCE_C_VECTOR_H
#define SOURCE_C_VECTOR_H

#include "../../../../../../../../../usr/include/c++/9/cmath"
#include "../../../../../../../../../usr/include/c++/9/cassert"
#include "../../../../../../../../../usr/include/c++/9/iostream"

// Definition of N_Tuple
template<size_t dimension, typename T>
struct N_Tuple
{
	T elements[dimension] = {};

	__device__ __host__ T &operator[](const size_t index)
	{
		return elements[index];
	}

	__device__ __host__ const T &operator[](const size_t index) const
	{
		return elements[index];
	}

	__device__ __host__ T norm()
	{
		T result{};
		for (size_t index = dimension; index--; result = *this * (*this));
		return std::sqrt(result);
	}
	__device__ __host__ N_Tuple<dimension, T> &normalize()
	{
		T norm = this->norm();
		*this = (*this) / norm;
		return *this;
	}
	// Vector addition
	__device__ __host__ N_Tuple<dimension, T> & operator+=(const N_Tuple<dimension, T> &rhs)
	{
		for (size_t index = dimension; index--; (*this)[index] += rhs[index]);
		return *this;
	}
	// Scalar multiplication
	__device__ __host__ N_Tuple<dimension, T> & operator*=(const T rhs)
	{
		for (size_t index = dimension; index--; (*this)[index] *= rhs);
		return *this;
	}


};

// Multiply N_Tuple with number
template<size_t dimension, typename T>
__device__ __host__ N_Tuple<dimension, T> operator*(const N_Tuple<dimension, T> &lhs, const T rhs)
{
	N_Tuple<dimension, T> result;
	for (size_t index = dimension; index--; result[index] = lhs[index] * rhs);
	return result;
}

template<size_t dimension, typename T>
__device__ __host__ N_Tuple<dimension, T> operator*(const T lhs, const N_Tuple<dimension, T> &rhs)
{
	return rhs * lhs;
}

template<size_t dimension, typename T>
__device__ __host__ N_Tuple<dimension, T> operator/(const N_Tuple<dimension, T> &lhs, const T rhs)
{
	N_Tuple<dimension, T> result;
	for (size_t index = dimension; index--; result[index] = lhs[index] / rhs);
	return result;
}

// Dot-product
template<size_t dimension, typename T>
__device__ __host__ T operator*(const N_Tuple<dimension, T> &lhs, const N_Tuple<dimension, T> &rhs)
{
	T result{};
	for (size_t index = dimension; index--; result += lhs[index] * rhs[index]);
	return result;
}

// Vector addition
template<size_t dimension, typename T>
__device__ __host__ N_Tuple<dimension, T> operator+(N_Tuple<dimension, T> lhs, const N_Tuple<dimension, T> &rhs)
{
	lhs += rhs;
	return lhs;
}


// Vector subtraction
template<size_t dimension, typename T>
__device__ __host__ N_Tuple<dimension, T> operator-(N_Tuple<dimension, T> lhs, const N_Tuple<dimension, T> &rhs)
{
	for (size_t index = dimension; index--; lhs[index] -= rhs[index]);
	return lhs;
}

// Invert direction_normalized
template<size_t dimension, typename T>
__device__ __host__ N_Tuple<dimension, T> operator-(const N_Tuple<dimension, T> &lhs)
{
	N_Tuple<dimension, T> result;
	for (size_t index = dimension; index--; result[index] = -lhs[index]);
	return result;
}

template<size_t dimension, typename T>
__device__ __host__ std::ostream &operator<<(std::ostream &out, const N_Tuple<dimension, T> &v)
{
	for (size_t index = 0; index < dimension; index++)
		out << v[index] << " ";
	return out;

}
using Vector3D =  N_Tuple<3, float>;
using Color =  N_Tuple<3, float>;
using Point3D =  N_Tuple<3, float>;


#endif //SOURCE_C_VECTOR_H

