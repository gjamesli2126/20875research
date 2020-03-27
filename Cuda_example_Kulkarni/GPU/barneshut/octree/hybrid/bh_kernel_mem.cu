/* -*- mode: c -*- */
/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/

#include <stdlib.h>
#include <stdio.h>

#include "../../../common/util_common.h"

#include "ptrtab.h"
#include "bh_kernel.h"
#include "bh_kernel_mem.h"

void allocate_gpu_tree_device(bh_oct_tree_gpu * h_root, bh_oct_tree_gpu ** d_root) {
	int i;
	
	bh_oct_tree_gpu * root;
	SAFE_MALLOC(root, sizeof(bh_oct_tree_gpu));

	// copy value params
	root->nnodes = h_root->nnodes;
	root->depth = h_root->depth;

	// allocate arrays on device
	CUDA_SAFE_CALL(cudaMalloc(&(root->nodes0), sizeof(gpu_node0)*root->nnodes));
	CUDA_SAFE_CALL(cudaMalloc(&(root->nodes1), sizeof(gpu_node1)*root->nnodes));
	CUDA_SAFE_CALL(cudaMalloc(&(root->nodes2), sizeof(gpu_node2)*root->nnodes));
	CUDA_SAFE_CALL(cudaMalloc(&(root->nodes3), sizeof(gpu_node3)*root->nnodes));

	#ifndef USE_LOCAL_STACK
	CUDA_SAFE_CALL(cudaMalloc(&(root->stk_nodes), sizeof(int)*root->depth*8*NUM_OF_THREADS_PER_BLOCK*NUM_OF_BLOCKS));
	CUDA_SAFE_CALL(cudaMalloc(&(root->stk_dsq), sizeof(float)*root->depth*8*NUM_OF_THREADS_PER_BLOCK*NUM_OF_BLOCKS));
	#endif

	*d_root = root;
}

void copy_gpu_tree_to_device(bh_oct_tree_gpu * h_root, bh_oct_tree_gpu * d_root) {
	CUDA_SAFE_CALL(cudaMemcpy(d_root->nodes0, h_root->nodes0, sizeof(gpu_node0)*h_root->nnodes, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_root->nodes1, h_root->nodes1, sizeof(gpu_node1)*h_root->nnodes, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_root->nodes2, h_root->nodes2, sizeof(gpu_node2)*h_root->nnodes, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_root->nodes3, h_root->nodes3, sizeof(gpu_node3)*h_root->nnodes, cudaMemcpyHostToDevice));
	
}

void copy_gpu_tree_to_host(bh_oct_tree_gpu * h_root, bh_oct_tree_gpu * d_root) {
	CUDA_SAFE_CALL(cudaMemcpy(h_root->nodes3, d_root->nodes3, sizeof(gpu_node3) * d_root->nnodes, cudaMemcpyDeviceToHost));
}

void free_gpu_tree_device(bh_oct_tree_gpu * d_root) {
	CUDA_SAFE_CALL(cudaFree(d_root->nodes0));
	CUDA_SAFE_CALL(cudaFree(d_root->nodes1));
	CUDA_SAFE_CALL(cudaFree(d_root->nodes2));
	CUDA_SAFE_CALL(cudaFree(d_root->nodes3));
	#ifndef USE_LOCAL_STACK
	CUDA_SAFE_CALL(cudaFree(d_root->stk_nodes));
	CUDA_SAFE_CALL(cudaFree(d_root->stk_dsq));
	#endif
}

void allocate_kernel_params(bh_kernel_params *params) {
	CUDA_SAFE_CALL(cudaMalloc(&(params->d_points_sorted), sizeof(bh_oct_tree_node*) * params->nbodies));
}

void copy_kernel_params_to_device(bh_kernel_params params) {
	// copy host data to device params memory locations
  CUDA_SAFE_CALL(cudaMemcpy(params.d_points_sorted, params.h_points_sorted, sizeof(bh_oct_tree_node*) * params.nbodies, cudaMemcpyHostToDevice));
}

void free_kernel_params_device(bh_kernel_params params) {
	CUDA_SAFE_CALL(cudaFree(params.d_points_sorted));
}
