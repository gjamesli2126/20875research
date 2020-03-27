/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include "pc_mem.h"

void alloc_tree_dev(gpu_tree *h_root, gpu_tree *d_root)
{
	CUDA_SAFE_CALL(cudaMalloc(&(d_root->nodes0), sizeof(gpu_node0)*h_root->max_nnodes));
	CUDA_SAFE_CALL(cudaMalloc(&(d_root->nodes1), sizeof(gpu_node1)*h_root->max_nnodes));
	CUDA_SAFE_CALL(cudaMalloc(&(d_root->nodes2), sizeof(gpu_node2)*h_root->max_nnodes));
	CUDA_SAFE_CALL(cudaMalloc(&(d_root->nodes3), sizeof(gpu_node3)*h_root->max_nnodes));
}

void alloc_set_dev(gpu_point_set *h_set, gpu_point_set *d_set)
{
	CUDA_SAFE_CALL(cudaMalloc(&(d_set->nodes0), sizeof(gpu_node0)*h_set->npoints));
	CUDA_SAFE_CALL(cudaMalloc(&(d_set->nodes3), sizeof(gpu_node3)*h_set->npoints));
}

/*void alloc_kernel_params_dev(pc_kernel_params *d_params)
{
	CUDA_SAFE_CALL(cudaMalloc(&(d_params->points), sizeof(int) * d_params->npoints));
}*/

void copy_tree_to_dev(gpu_tree *h_root, gpu_tree *d_root)
{
	CUDA_SAFE_CALL(cudaMemcpy(d_root->nodes0, h_root->nodes0, sizeof(gpu_node0)*h_root->max_nnodes, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_root->nodes1, h_root->nodes1, sizeof(gpu_node1)*h_root->max_nnodes, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_root->nodes2, h_root->nodes2, sizeof(gpu_node2)*h_root->max_nnodes, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_root->nodes3, h_root->nodes3, sizeof(gpu_node3)*h_root->max_nnodes, cudaMemcpyHostToDevice));
}

void copy_set_to_dev(gpu_point_set *h_set, gpu_point_set *d_set)
{
    CUDA_SAFE_CALL(cudaMemcpy(d_set->nodes0, h_set->nodes0, sizeof(gpu_node0)*h_set->npoints, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_set->nodes3, h_set->nodes3, sizeof(gpu_node3)*h_set->npoints, cudaMemcpyHostToDevice));
}

void copy_tree_to_host(gpu_tree *h_root, gpu_tree *d_root)
{
	CUDA_SAFE_CALL(cudaMemcpy(h_root->nodes3, d_root->nodes3, sizeof(gpu_node3)*h_root->max_nnodes, cudaMemcpyDeviceToHost));
#ifdef TRACK_TRAVERSALS
	CUDA_SAFE_CALL(cudaMemcpy(h_root->nodes0, d_root->nodes0, sizeof(gpu_node0)*h_root->max_nnodes, cudaMemcpyDeviceToHost));
#endif
}

void copy_set_to_host(gpu_point_set *h_set, gpu_point_set *d_set)
{
	CUDA_SAFE_CALL(cudaMemcpy(h_set->nodes3, d_set->nodes3, sizeof(gpu_node3)*h_set->npoints, cudaMemcpyDeviceToHost));
#ifdef TRACK_TRAVERSALS
	CUDA_SAFE_CALL(cudaMemcpy(h_set->nodes0, d_set->nodes0, sizeof(gpu_node0)*h_set->npoints, cudaMemcpyDeviceToHost));
#endif
}

/*void copy_kernel_params_to_dev(pc_kernel_params *h_params, pc_kernel_params *d_params)
{
	CUDA_SAFE_CALL(cudaMemcpy(d_params->points, h_params->points, sizeof(int)*d_params->npoints, cudaMemcpyHostToDevice));
}*/

void free_tree_dev(gpu_tree *d_root)
{
	CUDA_SAFE_CALL(cudaFree(d_root->nodes0));
	CUDA_SAFE_CALL(cudaFree(d_root->nodes1));
	CUDA_SAFE_CALL(cudaFree(d_root->nodes2));
	CUDA_SAFE_CALL(cudaFree(d_root->nodes3));
}

void free_set_dev(gpu_point_set *d_set)
{
	CUDA_SAFE_CALL(cudaFree(d_set->nodes0));
	CUDA_SAFE_CALL(cudaFree(d_set->nodes3));
}

/*void free_kernel_params_dev(pc_kernel_params *d_params) {
	CUDA_SAFE_CALL(cudaFree(d_params->points));
}*/


