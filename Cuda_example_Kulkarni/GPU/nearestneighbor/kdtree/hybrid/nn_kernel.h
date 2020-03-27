/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __KNN_BB_KERNEL_H_
#define __KNN_BB_KERNEL_H_
#include "nn_data_types.h"

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

#define LEFT cur_node2.items.left
#define RIGHT cur_node2.items.right

__global__ void init_kernel(void);
//__global__ void nearest_neighbor_search (gpu_tree gpu_tree, gpu_point *d_training_points, int n_training_points, gpu_point *d_search_points, int n_search_points);
__global__ void nearest_neighbor_search(kernel_params params);

gpu_tree * build_gpu_tree(KDCell * root);
int block_gpu_tree(KDCell * c_node, gpu_tree * h_root, int index, int depth);
void block_tree_info(gpu_tree * h_root, KDCell * c_node, int depth);
void free_gpu_tree(gpu_tree * root);

gpu_tree * gpu_transform_tree(KDCell *root);
void gpu_free_tree_host(gpu_tree *h_tree);

#endif
