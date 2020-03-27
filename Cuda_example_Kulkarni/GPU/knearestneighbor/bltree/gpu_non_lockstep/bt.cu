/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
//
//  main.cpp
//  BallTree-kNN

#include "bt_common.h"
#include "bt_functions.h"
#include "bt_kernel.h"

int sort_flag = 0;
int check_flag = 0;
int verbose_flag = 0;
int warp_flag = 0;
int ratio_flag = 0;
unsigned int npoints = 0;
unsigned int nsearchpoints = 0;

unsigned int K = 1;

datapoint *points = NULL;
datapoint *search_points = NULL;
float* nearest_distance = NULL;
int* nearest_point_index = NULL;

node* tree = NULL;
unsigned int max_depth = 0;
unsigned int nnodes = 0;

TIME_INIT(read_data);
TIME_INIT(build_tree);
TIME_INIT(sort);
TIME_INIT(kernel);
TIME_INIT(traversal);

int main(int argc, char * argv[]) {
    
    TIME_START(read_data);
    read_input(argc, argv);
    TIME_END(read_data);
    datapoint** dataList = new datapoint* [npoints];
    for (int i = 0; i < npoints; i ++) {
        dataList[i] = &points[i];
    }
    TIME_START(build_tree);
    tree = construct_tree(points, 0, npoints - 1, dataList, 0, 1);
    TIME_END(build_tree);
    printf("The max depth is %d, the nodes number is %d.\n", max_depth, nnodes);
//    printTree(tree, 0);
    
    TIME_START(sort);
    if (sort_flag) {
        sort_search_points(search_points, 0, nsearchpoints);
    }
    TIME_END(sort);
    
	init_kernel<<<1, 1>>>();
	gpu_tree * h_tree = gpu_transform_tree(tree);
	gpu_tree * d_tree = gpu_copy_to_dev(h_tree);

	// Allocate variables to store results of each thread
	datapoint * d_search_points;
	float * d_nearest_distance;
	int * d_nearest_point_index;
	// Read from but not written to
	CUDA_SAFE_CALL(cudaMalloc(&d_search_points, sizeof(datapoint)*nsearchpoints));
	CUDA_SAFE_CALL(cudaMemcpy(d_search_points, search_points, sizeof(datapoint)*nsearchpoints, cudaMemcpyHostToDevice));

	// Immediatly written to at kernel
	CUDA_SAFE_CALL(cudaMalloc(&d_nearest_distance, sizeof(float)*nsearchpoints*K));
	CUDA_SAFE_CALL(cudaMalloc(&d_nearest_point_index, sizeof(int)*nsearchpoints*K));

    TIME_START(traversal);
	//gpu_print_tree_host(h_tree);
	dim3 grid(NUM_THREAD_BLOCKS, 1, 1);
	dim3 block(NUM_THREADS_PER_BLOCK, 1, 1);
	unsigned int smem_bytes = DIM*NUM_THREADS_PER_BLOCK*sizeof(float) + K*NUM_THREADS_PER_BLOCK*sizeof(int) + K*NUM_THREADS_PER_BLOCK*sizeof(float);
	TIME_START(kernel);
    k_nearest_neighbor_search<<<grid, block, smem_bytes>>>(*d_tree, nsearchpoints, d_search_points, d_nearest_distance,
			d_nearest_point_index, K);

	cudaError_t err = cudaThreadSynchronize();
	if(err != cudaSuccess) {
		fprintf(stderr,"Kernel failed with error: %s\n", cudaGetErrorString(err));
		exit(1);
	}
    TIME_END(kernel);
//    for(int i = 0; i < nsearchpoints; i ++) {
//        k_nearest_neighbor_search(tree, &search_points[i], i*K);
//    }
	// Copy results back
	CUDA_SAFE_CALL(cudaMemcpy(nearest_point_index, d_nearest_point_index, sizeof(int)*nsearchpoints*K, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(nearest_distance, d_nearest_distance, sizeof(int)*nsearchpoints*K, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_nearest_point_index));
	CUDA_SAFE_CALL(cudaFree(d_nearest_distance));
	CUDA_SAFE_CALL(cudaFree(d_search_points));

    TIME_END(traversal);
    print_result();
    
    TIME_ELAPSED_PRINT(read_data, stdout);
    TIME_ELAPSED_PRINT(build_tree, stdout);
    TIME_ELAPSED_PRINT(sort, stdout);
    TIME_ELAPSED_PRINT(kernel, stdout);
    TIME_ELAPSED_PRINT(traversal, stdout);
    
    delete [] points;
    delete [] search_points;
    delete [] dataList;
    
    return 0;
}
