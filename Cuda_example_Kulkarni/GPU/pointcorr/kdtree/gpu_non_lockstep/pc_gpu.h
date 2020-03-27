/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __PC_GPU_H
#define __PC_GPU_H

#include <cuda.h>
#include "pc.h"

#define NUM_THREAD_BLOCKS 1024
#define THREADS_PER_BLOCK (6*32)

#define THREADS_PER_WARP 32

#define NWARPS_PER_BLOCK (THREADS_PER_BLOCK / THREADS_PER_WARP)
#define NWARPS (NUM_THREAD_BLOCKS*NWARPS_PER_BLOCK)

#define WARP_INDEX (threadIdx.x >> 5)
#define GLOBAL_WARP_INDEX (WARP_INDEX + (blockIdx.x*NWARPS_PER_BLOCK))
#define THREAD_INDEX_IN_WARP threadIdx.x & 0x1f

typedef struct _gpu_node0 {
	#ifdef TRACK_TRAVERSALS
	int nodes_accessed;
	#endif

	float coord_max[DIM];
	float min[DIM];
} gpu_node0;

typedef struct _gpu_node1 {
	int splitType; 
	int id;
} gpu_node1;

typedef struct _gpu_node2 {
	int left;
	int right;
} gpu_node2;

typedef struct _gpu_node3 {
	int corr;
	kd_cell *cpu_addr;	
} gpu_node3;

typedef struct _gpu_tree {
	gpu_node0 *nodes0;
	gpu_node1 *nodes1;
	gpu_node2 *nodes2;
	gpu_node3 *nodes3;

	unsigned int nnodes;
	unsigned int tree_depth;	

} gpu_tree;

typedef struct _pc_kernel_params {
	gpu_tree tree;
	float rad;
	int npoints;
	int *points; // index value of input points

} pc_kernel_params;

/* Kernel Macros */

#define STACK_INIT() sp = 1; stack[1] = 0
#define STACK_POP() sp -= 1
#define STACK_PUSH() sp += 1  
#define STACK_TOP_NODE_INDEX stack[sp]

#define CUR_NODE0 cur_node0
#define CUR_NODE1 cur_node1
#define CUR_NODE2 cur_node2


#endif
