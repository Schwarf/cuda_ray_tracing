//
// Created by andreas on 20.12.21.
//

#ifndef CUDAMEMORY_CUH
#define CUDAMEMORY_CUH
#include <iostream>
#include <vector>

#define checkCudaErrors(value) check_cuda( (value), #value, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
				  file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

template<typename T>
class CUDAMemory
{
public:
	static void allocate_managed_instance(T * &p_data_pointer, size_t size)
	{
		checkCudaErrors( cudaMallocManaged((void **)&p_data_pointer, sizeof(T)*size));
	}
	static void allocate_instance(T * &p_data_pointer, size_t size)
	{
		checkCudaErrors( cudaMalloc((void **)&p_data_pointer, sizeof(T)*size));
	}

	static void allocate_managed_pointer_to_instance(T ** &p_data_pointer, size_t size)
	{
		checkCudaErrors( cudaMallocManaged((void **)&p_data_pointer, sizeof(T*)*size));
	}
	static void allocate_pointer_to_instance(T ** &p_data_pointer, size_t size)
	{
		checkCudaErrors( cudaMalloc((void **)&p_data_pointer, sizeof(T*)*size));
	}


	static void copy_from_host_vector_to_device(std::vector<T> & origin, T * &target, size_t size)
	{
		checkCudaErrors(cudaMemcpy(target, origin.data(), sizeof(T)*size, cudaMemcpyHostToDevice));
	}
	static void copy_from_device_to_host_vector(std::vector<T> & target, T * &origin, size_t size)
	{
		checkCudaErrors(cudaMemcpy(target.data(), origin, sizeof(T)*size, cudaMemcpyDeviceToHost));
		}

	static void release(T * &p_data_pointer)
	{
		cudaFree(p_data_pointer);
	}
};


#endif //CUDAMEMORY_CUH
