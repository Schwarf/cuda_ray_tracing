//
// Simpl
// Created by andreas on 02.10.21.
//

#ifndef SOURCE_C_VECTOR_H
#define SOURCE_C_VECTOR_H

#include "../../../../../../../../../usr/include/c++/9/cmath"
#include "../../../../../../../../../usr/include/c++/9/cassert"
#include "../../../../../../../../../usr/include/c++/9/iostream"

// Definition of c_vector
template<size_t dimension, typename T>
struct c_vector
{
	T elements[dimension] = {};

	__host__ __device__ T &operator[](const size_t index)
	{
		return elements[index];
	}

	__host__ __device__ const T &operator[](const size_t index) const
	{
		return elements[index];
	}

	__host__ __device__ T norm()
	{
		T result{};
		for (size_t index = dimension; index--; result = *this * (*this));
		return std::sqrt(result);
	}
	__host__ __device__ c_vector<dimension, T> &normalize()
	{
		T norm = this->norm();
		*this = (*this) / norm;
		return *this;
	}

};

// Multiply c_vector with number
template<size_t dimension, typename T>
__host__ __device__ c_vector<dimension, T> operator*(const c_vector<dimension, T> &lhs, const T rhs)
{
	c_vector<dimension, T> result;
	for (size_t index = dimension; index--; result[index] = lhs[index] * rhs);
	return result;
}

template<size_t dimension, typename T>
__host__ __device__ c_vector<dimension, T> operator*(const T lhs, const c_vector<dimension, T> &rhs)
{
	return rhs * lhs;
}

template<size_t dimension, typename T>
__host__ __device__ c_vector<dimension, T> operator/(const c_vector<dimension, T> &lhs, const T rhs)
{
	c_vector<dimension, T> result;
	for (size_t index = dimension; index--; result[index] = lhs[index] / rhs);
	return result;
}

// Dot-product
template<size_t dimension, typename T>
__host__ __device__ T operator*(const c_vector<dimension, T> &lhs, const c_vector<dimension, T> &rhs)
{
	T result{};
	for (size_t index = dimension; index--; result += lhs[index] * rhs[index]);
	return result;
}

// Vector addition
template<size_t dimension, typename T>
__host__ __device__ c_vector<dimension, T> operator+(c_vector<dimension, T> lhs, const c_vector<dimension, T> &rhs)
{
	for (size_t index = dimension; index--; lhs[index] += rhs[index]);
	return lhs;
}

// Vector subtraction
template<size_t dimension, typename T>
__host__ __device__ c_vector<dimension, T> operator-(c_vector<dimension, T> lhs, const c_vector<dimension, T> &rhs)
{
	for (size_t index = dimension; index--; lhs[index] -= rhs[index]);
	return lhs;
}

// Invert direction_normalized
template<size_t dimension, typename T>
__host__ __device__ c_vector<dimension, T> operator-(const c_vector<dimension, T> &lhs)
{
	c_vector<dimension, T> result;
	for (size_t index = dimension; index--; result[index] = -lhs[index]);
	return result;
}

typedef c_vector<3, float> c_vector3;

template<size_t dimension, typename T>
__host__ __device__ std::ostream &operator<<(std::ostream &out, const c_vector<dimension, T> &v)
{
	for (size_t index = 0; index < dimension; index++)
		out << v[index] << " ";
	return out;

}

#endif //SOURCE_C_VECTOR_H
