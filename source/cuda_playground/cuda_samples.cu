//
// Created by andreas on 24.10.21.
//

// This program computes the sum of two vectors of length N
// By: Nick from CoffeeBeforeArch

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include "vector_add.cu"
#include <fstream>
#include <ctime>

#define checkCudaErrors(value) check_cuda( (value), #value, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const function, const char * const file, int const line)
{
	if(result)
	{
		std::cerr<< "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << function << "' \n";
		cudaDeviceReset();
		exit(99);
	}
}


__global__ void render(float * buffer, int width, int height)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if( (i > width-1) || (j > height-1))
	{
		return;
	}
	int pixel_index = j*width*3  + i*3;
	buffer[pixel_index] = 0.2f;
	buffer[pixel_index + 1] = 0.7f;
	buffer[pixel_index + 2] = 0.8f;
}

// CUDA kernel for vector addition
// __global__ means this is called from the CPU, and runs on the GPU
__global__ void vectorAdd(int *__restrict__ a, int *__restrict__ b,
						  int *__restrict__ c, int N) {
	// Calculate global thread ID
	//int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	IntegerVectorAddition add;
	add.set_summand_one(a, N);
	add.set_summand_two(b, N);

	// Boundary check
	//if (tid < N) c[tid] = a[tid] + b[tid];
	add.add(c);
}

// Check vector add result
void verify_result(std::vector<int> &a, std::vector<int> &b,
				   std::vector<int> &c) {
	long long int sum = 0;
	for (int i = 0; i < a.size(); i++) {
		assert(c[i] == a[i] + b[i]);
		sum += c[i];
	}
	printf("Sum = %lld \n", sum);
}

int main() {
	// Array size of 2^16 (65536 elements)
	constexpr int N = 1 << 16;
	constexpr size_t bytes = sizeof(int) * N;

	// Vectors for holding the host-side (CPU-side) data
	std::vector<int> a;
	a.reserve(N);
	std::vector<int> b;
	b.reserve(N);
	std::vector<int> c;
	c.reserve(N);

	// Initialize random numbers in each array
	for (int i = 0; i < N; i++) {
		a.push_back(rand() % 100);
		b.push_back(rand() % 100);
	}

	// Allocate memory on the device
	int *d_a, *d_b, *d_c;
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	// Copy data from the host to the device (CPU -> GPU)
	cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

	// Threads per CTA (1024)
	int NUM_THREADS = 1 << 10;

	// CTAs per Grid
	// We need to launch at LEAST as many threads as we have elements
	// This equation pads an extra CTA to the grid if N cannot evenly be divided
	// by NUM_THREADS (e.g. N = 1025, NUM_THREADS = 1024)
	int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

	// Launch the kernel on the GPU
	// Kernel calls are asynchronous (the CPU program continues execution after
	// call, but no necessarily before the kernel finishes)
	vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);

	// Copy sum vector from device to host
	// cudaMemcpy is a synchronous operation, and waits for the prior kernel
	// launch to complete (both go to the default stream in this case).
	// Therefore, this cudaMemcpy acts as both a memcpy and synchronization
	// barrier.
	cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

	// Check result for errors
	verify_result(a, b, c);

	// Free memory on device
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	std::cout << "COMPLETED SUCCESSFULLY\n";

	int width =1024;
	int height = 768;
	int thread_x =8;
	int thread_y =8;
	int number_of_pixels = width*height;
	size_t buffer_size = 3*number_of_pixels*sizeof(float);
	float *buffer =nullptr;
	checkCudaErrors(cudaMallocManaged((void **)&buffer, buffer_size));
	clock_t start, stop;
	start = clock();
	dim3 blocks(width/thread_x +1, height/thread_y+1);
	dim3 threads(thread_x, thread_y);
	render<<<blocks, threads>>>(buffer, width, height);
	checkCudaErrors(cudaGetLastError());
	cudaDeviceSynchronize();
	stop =clock();
	double seconds = ((double) (stop-start))/CLOCKS_PER_SEC;
	std::cerr << "took " << seconds << " seconds. \n";
	std::ofstream ofs;
	ofs.open("./cuda_image.ppm");
	ofs << "P6\n" << width << " " << height << "\n255\n";
	for(int j = height-1; j >=0; --j)
	{
		for(int i = 0; i <width; ++i)
		{
			size_t pixel_index = j*3*width + 3*i;
			char red = static_cast<char>(std::max(0.f, std::min(1.f, buffer[pixel_index]))*255);
			char green = static_cast<char>(std::max(0.f, std::min(1.f, buffer[pixel_index+1]))*255);
			char blue = static_cast<char>(std::max(0.f, std::min(1.f, buffer[pixel_index+2]))*255);
			//std::cout << red << " " << green << " " << blue << std::endl;
			ofs << red << " " << green << " " << blue << " ";
		}
	}
	ofs.close();
	cudaFree(buffer);
	return 0;
}

