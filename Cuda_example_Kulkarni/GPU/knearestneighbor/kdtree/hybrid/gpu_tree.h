/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __GPU_TREE_H_
#define __GPU_TREE_H_

#include <limits.h>
#include "knn.h"

#define NULL_NODE -1

#define NOTHING_DONE (0)
#define INITIAL_VISIT (1)
#define NEAR_DONE (2)
#define FAR_DONE (3)

typedef union {
	long long __val;
	struct {
		int axis;
		int point_index;
		// added by Cambridge
		int depth;
		int pre_id;
	} items;
} gpu_tree_node_0;

typedef struct gpu_tree_node_1_ {
	int left;
	int right;
} gpu_tree_node_1;

typedef struct gpu_tree_node_2_ {
	float point[DIM];
} gpu_tree_node_2;

typedef union _stack_entry {
	long long __val;
	struct {
		//unsigned int phase : 2;
		int jump_node;
		float axis_dist;
	} items;
} stack_entry;

typedef struct _gpu_tree {

	gpu_tree_node_0 * nodes0;
	gpu_tree_node_1 * nodes1;
	gpu_tree_node_2 * nodes2;
	
	int *stk_node;
	float *stk_axis_dist;

	int nnodes;
	int max_depth;

} gpu_tree;


#endif
