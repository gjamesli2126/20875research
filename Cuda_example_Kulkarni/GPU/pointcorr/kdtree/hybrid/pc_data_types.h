/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __POINT_CORR_DATA_TYPES_H_
#define __POINT_CORR_DATA_TYPES_H_

#include "../../../common/util_common.h"
//#ifndef TRACK_TRAVERSALS
//#define TRACK_TRAVERSALS
//#endif

#ifndef DIM
#define DIM 7
#endif

#ifndef RADIUS
#define RADIUS (0.03f)
#endif

#ifndef SPLICE_DEPTH
#define SPLICE_DEPTH 2
#endif

#define SPLIT_LEAF (DIM)

typedef struct _kd_cell
{
	int id;
	int corr;
	int splitType;
	int depth;
	int pre_id;
	float coord_max[DIM];
	float min[DIM];
	struct _kd_cell *left;
	struct _kd_cell *right;

	#ifdef TRACK_TRAVERSALS
	int nodes_accessed;
	int nodes_truncated;
	#endif
} kd_cell;

union coord_pair{
	long __val;
	struct {
		float max;
		float min;
	} items;
};

typedef struct _gpu_node0 {
	union coord_pair coord[DIM];
#ifdef TRACK_TRAVERSALS
    int nodes_accessed;
    int nodes_truncated;
#endif
} gpu_node0;

typedef struct _gpu_node1
{
	int splitType;
    int my_index;
	int depth;
	int pre_id;
} gpu_node1;

typedef struct _gpu_node2
{
	int left;
	int right;
} gpu_node2;

typedef struct _gpu_node3
{
	int corr;
	int point_id;
	//	kd_cell *cpu_addr;
} gpu_node3;

typedef struct _gpu_tree
{
	gpu_node0 *nodes0;
	gpu_node1 *nodes1;
	gpu_node2 *nodes2;
	gpu_node3 *nodes3;

	unsigned int nnodes;
	unsigned int max_nnodes;
	unsigned int tree_depth;
} gpu_tree;

typedef struct _gpu_point_set
{
    unsigned int npoints;
	gpu_node0 *nodes0;
	gpu_node3 *nodes3;
} gpu_point_set;

typedef struct _pc_kernel_params {
	gpu_tree tree;
	int root_index;
    gpu_point_set set;
	float rad;
	int npoints;
	int* index_buffer;
} pc_kernel_params;

typedef struct _pc_pre_kernel_params {
    gpu_tree tree;
    int root_index;
    gpu_point_set set;
    bool *relation_matrix;
    float rad;
    int npoints;
    int tree_max_nnodes;
} pc_pre_kernel_params;

/* Cuda Macros */
#ifndef NUM_OF_WARPS_PER_BLOCK
#define NUM_OF_WARPS_PER_BLOCK 			6
#endif 

#ifndef NUM_OF_BLOCKS
#define NUM_OF_BLOCKS 					(1024)
#endif 

#define NUM_OF_THREADS_PER_WARP 		32
#define NUM_OF_THREADS_PER_BLOCK 		NUM_OF_WARPS_PER_BLOCK * NUM_OF_THREADS_PER_WARP

#define WARP_INDEX (threadIdx.x >> 5)
#define GLOBAL_WARP_INDEX (WARP_INDEX + (blockIdx.x*NWARPS_PER_BLOCK))
#define THREAD_INDEX_IN_WARP threadIdx.x & 0x1f

/* Kernel Macros */

#ifdef USE_SMEM
#define STACK_INIT() sp = 1; stack[WARP_INDEX][1] = 0
//mask[WARP_INDEX][1] = 0xffffffff
#define STACK_POP() sp -= 1
#define STACK_PUSH() sp += 1
#define STACK_TOP_NODE_INDEX stack[WARP_INDEX][sp]
#define STACK_TOP_MASK mask[WARP_INDEX][sp]
#define CUR_NODE0 cur_node0[WARP_INDEX]
#define CUR_NODE1 cur_node1[WARP_INDEX]
#define CUR_NODE2 cur_node2[WARP_INDEX]
#else
#define STACK_INIT() sp = 1; stack[1] = 0
//mask[1] = 0xffffffff
#define STACK_POP() sp -= 1
#define STACK_PUSH() sp += 1
#define STACK_TOP_NODE_INDEX stack[sp]
#define STACK_TOP_MASK mask[sp]
#define CUR_NODE0 cur_node0
#define CUR_NODE1 cur_node1
#define CUR_NODE2 cur_node2
#endif

#endif
