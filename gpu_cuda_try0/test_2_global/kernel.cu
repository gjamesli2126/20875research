#include<stdio.h>
#include<stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define size 32

__global__
void VectorAdd(int* a, int* b, int* c, int n) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		c[i] = a[i] + b[i];
		
	}
}

void saysomething(int type) {
	 printf("wow__well don,,,%d",type);
}

int main() {

	int* gpu_a, * gpu_b, * gpu_c;
	int i;

	cudaMallocManaged(&gpu_a, sizeof(int) * size);
	cudaMallocManaged(&gpu_b, sizeof(int) * size);
	cudaMallocManaged(&gpu_c, sizeof(int) * size);
	for (i = 0; i < size; i++) {
		gpu_a[i] = i;
		gpu_b[i] = 2 * i;
	}
	saysomething(0);
	VectorAdd << <2,16 >> > (gpu_a, gpu_b, gpu_c, size);// number of thread block , number of thread in each threadblock
	// number of thread block x number of thread in each threadblock=total size
	cudaDeviceSynchronize();
	for (i = 0; i < size; i++) {
		printf("c[%d]=%d\n", i, gpu_c[i]);
		//printf(".");
	}
	cudaFree(gpu_a);
	cudaFree(gpu_b);
	cudaFree(gpu_c);

			

	return size;
}
