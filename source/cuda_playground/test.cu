//
// Created by andreas on 02.11.21.
//

#include <iostream>
#include <time.h>
#ifndef VEC3H
#define VEC3H
#include <stdlib.h>

class vec3  {


public:
	__device__ __host__ vec3() {}
	__device__ __host__ vec3(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2; }
	__device__ __host__ inline float x() const { return e[0]; }
	__device__ __host__ inline float y() const { return e[1]; }
	__device__ __host__ inline float z() const { return e[2]; }
	__device__ __host__ inline float r() const { return e[0]; }
	__device__ __host__ inline float g() const { return e[1]; }
	__device__ __host__ inline float b() const { return e[2]; }

	__device__ __host__ inline const vec3& operator+() const { return *this; }
	__device__ __host__ inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
	__device__ __host__ inline float operator[](int i) const { return e[i]; }
	__device__ __host__ inline float& operator[](int i) { return e[i]; };

	__device__ __host__ inline vec3& operator+=(const vec3 &v2);
	__device__ __host__ inline vec3& operator-=(const vec3 &v2);
	__device__ __host__ inline vec3& operator*=(const vec3 &v2);
	__device__ __host__ inline vec3& operator/=(const vec3 &v2);
	__device__ __host__ inline vec3& operator*=(const float t);
	__device__ __host__ inline vec3& operator/=(const float t);

	__device__ __host__ inline float length() const { return sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]); }
	__device__ __host__ inline float squared_length() const { return e[0]*e[0] + e[1]*e[1] + e[2]*e[2]; }
	__device__ __host__ inline void make_unit_vector();


	float e[3];
};



inline std::istream& operator>>(std::istream &is, vec3 &t) {
	is >> t.e[0] >> t.e[1] >> t.e[2];
	return is;
}

inline std::ostream& operator<<(std::ostream &os, const vec3 &t) {
	os << t.e[0] << " " << t.e[1] << " " << t.e[2];
	return os;
}

__device__ __host__ inline void vec3::make_unit_vector() {
	float k = 1.0 / sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
	e[0] *= k; e[1] *= k; e[2] *= k;
}

__device__ __host__ inline vec3 operator+(const vec3 &v1, const vec3 &v2) {
	return vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__device__ __host__ inline vec3 operator-(const vec3 &v1, const vec3 &v2) {
	return vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__device__ __host__ inline vec3 operator*(const vec3 &v1, const vec3 &v2) {
	return vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__device__ __host__ inline vec3 operator/(const vec3 &v1, const vec3 &v2) {
	return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__device__ __host__ inline vec3 operator*(float t, const vec3 &v) {
	return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__device__ __host__ inline vec3 operator/(vec3 v, float t) {
	return vec3(v.e[0]/t, v.e[1]/t, v.e[2]/t);
}

__device__ __host__ inline vec3 operator*(const vec3 &v, float t) {
	return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__device__ __host__ inline float dot(const vec3 &v1, const vec3 &v2) {
	return v1.e[0] *v2.e[0] + v1.e[1] *v2.e[1]  + v1.e[2] *v2.e[2];
}

__device__ __host__ inline vec3 cross(const vec3 &v1, const vec3 &v2) {
	return vec3( (v1.e[1]*v2.e[2] - v1.e[2]*v2.e[1]),
				 (-(v1.e[0]*v2.e[2] - v1.e[2]*v2.e[0])),
				 (v1.e[0]*v2.e[1] - v1.e[1]*v2.e[0]));
}


__device__ __host__ inline vec3& vec3::operator+=(const vec3 &v){
	e[0]  += v.e[0];
	e[1]  += v.e[1];
	e[2]  += v.e[2];
	return *this;
}

__device__ __host__ inline vec3& vec3::operator*=(const vec3 &v){
	e[0]  *= v.e[0];
	e[1]  *= v.e[1];
	e[2]  *= v.e[2];
	return *this;
}

__device__ __host__ inline vec3& vec3::operator/=(const vec3 &v){
	e[0]  /= v.e[0];
	e[1]  /= v.e[1];
	e[2]  /= v.e[2];
	return *this;
}

__device__ __host__ inline vec3& vec3::operator-=(const vec3& v) {
	e[0]  -= v.e[0];
	e[1]  -= v.e[1];
	e[2]  -= v.e[2];
	return *this;
}

__device__ __host__ inline vec3& vec3::operator*=(const float t) {
	e[0]  *= t;
	e[1]  *= t;
	e[2]  *= t;
	return *this;
}

__device__ __host__ inline vec3& vec3::operator/=(const float t) {
	float k = 1.0/t;

	e[0]  *= k;
	e[1]  *= k;
	e[2]  *= k;
	return *this;
}

__device__ __host__ inline vec3 unit_vector(vec3 v) {
	return v / v.length();
}

#endif


// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
				  file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

__global__ void render(vec3 *fb, int max_x, int max_y) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j*max_x + i;
	fb[pixel_index] = vec3( float(i) / max_x, float(j) / max_y, 0.2f);
}

int main() {
	int nx = 1200;
	int ny = 600;
	int tx = 8;
	int ty = 8;

	std::cerr << "Rendering a " << nx << "x" << ny << " image ";
	std::cerr << "in " << tx << "x" << ty << " blocks.\n";

	int num_pixels = nx*ny;
	size_t fb_size = num_pixels*sizeof(vec3);

	// allocate FB
	vec3 *fb;
	checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

	clock_t start, stop;
	start = clock();
	// Render our buffer
	dim3 blocks(nx/tx+1,ny/ty+1);
	dim3 threads(tx,ty);
	render<<<blocks, threads>>>(fb, nx, ny);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timer_seconds << " seconds.\n";

	// Output FB as Image
	std::cout << "P3\n" << nx << " " << ny << "\n255\n";
	for (int j = ny-1; j >= 0; j--) {
		for (int i = 0; i < nx; i++) {
			size_t pixel_index = j*nx + i;
			int ir = int(255.99*fb[pixel_index].r());
			int ig = int(255.99*fb[pixel_index].g());
			int ib = int(255.99*fb[pixel_index].b());
			std::cout << ir << " " << ig << " " << ib << "\n";
		}
	}

	checkCudaErrors(cudaFree(fb));
	float3 x{1,1,1};
	float3 y{2,2,2};
	auto result = x*y;
	std::cout <<
}
