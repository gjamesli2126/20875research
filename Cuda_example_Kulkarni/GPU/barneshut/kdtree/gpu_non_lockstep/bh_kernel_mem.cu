/* -*- mode: c -*- */
/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include "harness.h"

#include <stdlib.h>
#include <stdio.h>

#include "util_common.h"

#include "bh_kernel.h"
#include "bh_kernel_mem.h"

void allocate_gpu_tree_device(bh_gpu_tree * h_root, bh_gpu_tree ** d_root) {
	int i;
	
	bh_gpu_tree * root;
	SAFE_MALLOC(root, sizeof(bh_gpu_tree));

	// copy value params
	root->nnodes = h_root->nnodes;
	root->depth = h_root->depth;

	// allocate arrays on device
	CUDA_SAFE_CALL(cudaMalloc(&(root->nodes0), sizeof(gpu_node0)*root->nnodes));
	CUDA_SAFE_CALL(cudaMalloc(&(root->nodes1), sizeof(gpu_node1)*root->nnodes));
	CUDA_SAFE_CALL(cudaMalloc(&(root->nodes2), sizeof(gpu_node2)*root->nnodes));
	CUDA_SAFE_CALL(cudaMalloc(&(root->nodes3), sizeof(gpu_node3)*root->nnodes));

	*d_root = root;
}

void copy_gpu_tree_to_device(bh_gpu_tree * h_root, bh_gpu_tree * d_root) {
	CUDA_SAFE_CALL(cudaMemcpy(d_root->nodes0, h_root->nodes0, sizeof(gpu_node0)*h_root->nnodes, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_root->nodes1, h_root->nodes1, sizeof(gpu_node1)*h_root->nnodes, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_root->nodes2, h_root->nodes2, sizeof(gpu_node2)*h_root->nnodes, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_root->nodes3, h_root->nodes3, sizeof(gpu_node3)*h_root->nnodes, cudaMemcpyHostToDevice));
	
}

void copy_gpu_tree_to_host(bh_gpu_tree * h_root, bh_gpu_tree * d_root) {
	CUDA_SAFE_CALL(cudaMemcpy(h_root->nodes3, d_root->nodes3, sizeof(gpu_node3) * d_root->nnodes, cudaMemcpyDeviceToHost));
}

void free_gpu_tree_device(bh_gpu_tree * d_root) {
	CUDA_SAFE_CALL(cudaFree(d_root->nodes0));
	CUDA_SAFE_CALL(cudaFree(d_root->nodes1));
	CUDA_SAFE_CALL(cudaFree(d_root->nodes2));
	CUDA_SAFE_CALL(cudaFree(d_root->nodes3));
}
