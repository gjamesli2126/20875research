/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __NN_GPU_H
#define __NN_GPU_H

#include "knn.h"
#include "gpu_tree.h"

#define DOWN 0
#define UP 1

#ifndef NUM_OF_WARPS_PER_BLOCK
#define NUM_OF_WARPS_PER_BLOCK 			6
#endif 

#ifndef NUM_OF_BLOCKS
#define NUM_OF_BLOCKS 					(1024)
#endif 

#define NUM_OF_THREADS_PER_WARP 		32
#define NUM_OF_THREADS_PER_BLOCK 		NUM_OF_WARPS_PER_BLOCK * NUM_OF_THREADS_PER_WARP

#define WARP_INDEX (threadIdx.x >> 5)
#define GLOBAL_WARP_IDX (WARP_INDEX + (blockIdx.x*NUM_WARPS_PER_BLOCK))
#define THREAD_IDX_IN_WARP (threadIdx.x - (WARP_SIZE * WARP_INDEX))
#define IS_FIRST_THREAD_IN_WARP (threadIdx.x == (WARP_INDEX * WARP_SIZE))

// top levelfunctions
__global__ void init_kernel(void);
__global__ void nearest_neighbor_search(gpu_tree gpu_tree, int nsearchpoints, node * search_points, float * nearest_distance, int * nearest_point_index, int K, int* index_buffer
#ifdef TRACK_TRAVERSALS
																				,int *d_nodes_accessed
#endif																				
);

__global__ void nearest_neighbor_pre_search(gpu_tree gpu_tree, int nsearchpoints, node * search_points, float * nearest_distance, int * nearest_point_index, int K, int* d_matrix, int start, int end, int interval
#ifdef TRACK_TRAVERSALS
																				,int *d_nodes_accessed
#endif																				
);

gpu_tree * gpu_transform_tree(node *root);
gpu_tree * gpu_copy_to_dev(gpu_tree *h_tree);
void gpu_copy_tree_to_host(gpu_tree *d_tree, gpu_tree *h_tree);
void gpu_free_tree_dev(gpu_tree *d_tree);
void gpu_free_tree_host(gpu_tree *h_tree);
void gpu_print_tree_host(gpu_tree *h_tree);

#endif
