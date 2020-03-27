#include<stdio.h>
#include<stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <time.h>
#define size 40960000*2

__global__ 
void VectorAdd(int* a, int* b, int* c, int n) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		c[i] = a[i] + b[i];
	}
	
}
void NormalAdd(int* a, int* b, int* c, int n) {
	int i;
	for (i = 0; i < n; i++) {
		c[i] = a[i] + b[i];
	}
}

int main() {
	clock_t start;
	clock_t gputime, cputime;
	int *gpu_a, *gpu_b, *gpu_c;
	int* cpu_a, * cpu_b, * cpu_c;
	int i;
	int block_size, num_block;
	cudaMallocManaged(&gpu_a, sizeof(int) * size);
	cudaMallocManaged(&gpu_b, sizeof(int) * size);
	cudaMallocManaged(&gpu_c, sizeof(int) * size);
	for (i = 0; i < size; i++) {
		gpu_a[i] = i;
		gpu_b[i] = 2 * i;
	}
	cpu_a = (int*)malloc(sizeof(int) * size);
	cpu_b = (int*)malloc(sizeof(int) * size);
	cpu_c = (int*)malloc(sizeof(int) * size);
	for (i = 0; i < size; i++) {
		cpu_a[i] = i;
		cpu_b[i] = 2 * i;
	}
	start = clock();
	NormalAdd(cpu_a, cpu_b, cpu_c, size);
	cputime = clock() - start;
	free(cpu_a);
	free(cpu_b);
	free(cpu_c);

	//a = (int*)malloc(sizeof(int) * size);
	//b = (int*)malloc(sizeof(int) * size);
	//c = (int*)malloc(sizeof(int) * size);
	//write file
	FILE* f = fopen("try_thread_block.txt", "w");
	for (num_block = 1; num_block <= 16; num_block++) {
		for (block_size = 1; block_size <= 1024; block_size++) {
		
			//main code
			start = clock();
			VectorAdd <<<num_block, block_size >>> (gpu_a, gpu_b, gpu_c, size);//block number, thread in threadblock
			gputime = clock() - start;
			/*printf("GPU-----%f\n\n", (float)gputime);*/
			//print
			cudaDeviceSynchronize();
			//for (i = 0; i < size; i++) {
			//	//printf("c[%d]=%d\n", i, &c[i]);
			//	printf(".");
			//}
			//cudaFree(a);
			//cudaFree(b);
			//cudaFree(c);
			///------------------------------------------------------------------------------CPU

			//main code
			//start = clock();
			//NormalAdd(a, b, c, size);
			//cputime = clock() - start;
			//printf("CPU-----%f\n\n", (float)cputime);
			//for (i = 0; i < size; i++) {
			//	//printf("c[%d]=%d\n", i, &c[i]);
			//	printf(",");
			//}

			//printf("Block number: %d\tBlock Size: %d\tcpu: %.1f\tgpu: %.1f\tspeed : %.f r/ms\n", num_block, block_size , (float)cputime, (float)gputime, (float)size / (float)gputime);
			fprintf_s(f,"Block number: %d\tBlock Size: %d\tcpu: %.1f\tgpu: %.1f\tspeed : %.f r/ms\n", num_block, block_size, (float)cputime, (float)gputime, (float)size/(float)gputime);
		}
		printf("Block number: %d\tBlock Size: %d\tcpu: %.1f\tgpu: %.1f\tspeed : %.f r/ms\n", num_block, block_size-1, (float)cputime, (float)gputime, (float)size / (float)gputime);
	}
	fclose(f);
	return size;
}