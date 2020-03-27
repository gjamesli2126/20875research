/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
//
//  BBT_kenrel.hpp
//  BallTree-kNN

#ifndef __BT_KERNEL_H_
#define __BT_KERNEL_H_

#include "bt_functions.h"
#include "bt_gpu_tree.h"

extern neighbor *nearest_neighbor;

#define STACK_NODE stk_node_top
//#define POINT cur_node2.point
#define POINT_INDEX cur_node1.idx
//cur_node2.point[s]
#define LEFT cur_node1.left
#define RIGHT cur_node1.right

#define STACK_INIT() \
	sp = 0;	\
	stk_node = &gpu_tree.stk_node[gpu_tree.depth*2*blockIdx.x*blockDim.x + threadIdx.x]; \
	stk_node_top = 0;

#define STACK_PUSH() sp = sp + 1; \
	*stk_node = stk_node_top; \
	stk_node += blockDim.x; \

#define STACK_POP() sp = sp - 1; \
	stk_node -= blockDim.x; \
	if(sp >= 0) { \
	stk_node_top = *stk_node; \
	}




__global__ void init_kernel(void);
void k_nearest_neighbor_search(node* node, datapoint* point, int pos);

__global__ void k_nearest_neighbor_search (gpu_tree gpu_tree, int nsearchpoints, datapoint *d_search_points,
											float *d_nearest_distance, int *d_nearest_point_index, int K);

#endif /* BBT_kenrel_hpp */
