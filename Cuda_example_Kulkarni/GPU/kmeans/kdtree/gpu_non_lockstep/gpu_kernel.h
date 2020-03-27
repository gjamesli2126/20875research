/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __KMEANS_GPU_KERNEL_H__
#define __KMEANS_GPU_KERNEL_H__
#include "common.h"

#ifndef NUM_OF_WARPS_PER_BLOCK
#define NUM_OF_WARPS_PER_BLOCK 			6
#endif 

#ifndef NUM_OF_BLOCKS
#define NUM_OF_BLOCKS 					(128)
#endif 

#define NUM_OF_THREADS_PER_WARP 		32
#define NUM_OF_THREADS_PER_BLOCK 		NUM_OF_WARPS_PER_BLOCK * NUM_OF_THREADS_PER_WARP

#define WARP_INDEX (threadIdx.x >> 5)

#define STACK_INIT() sp = 1; \
	stk[sp] = 0;
#define STACK_POP() sp -= 1; 
#define STACK_PUSH() sp += 1; 
#define STACK_TOP_NODE_INDEX stk[sp]

__global__ void init_kernel(void);
__global__ void nearest_cluster (gpu_tree gpu_tree, DataPoint *points, int npoints, int K);

#endif
