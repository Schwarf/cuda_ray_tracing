
#pragma once

__device__ class IntegerVectorAddition
{
public:
	IntegerVectorAddition() = default;
	__device__ void set_summand_one(int *__restrict__ summand_one, int size_one)
	{
		summand_one_ = summand_one;
		size_one_ = size_one;
	}
	__device__ void set_summand_two(int *__restrict__ summand_two, int size_two)
	{
		summand_two_ = summand_two;
		size_two_ = size_two;
	}
	__device__ void add(int *__restrict__ result)
	{
		if (size_one_ != size_two_ || size_one_ ==0)
		{
			return;
		}
		int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;

		// Boundary check
		if (thread_id > size_one_)
		{
			return;
		}
		result[thread_id] = summand_one_[thread_id] + summand_two_[thread_id];
	}
	private:

	int *summand_one_;

	int *summand_two_;

	int size_one_;

	int size_two_;

};


