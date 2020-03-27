/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __GPU_TREE_H
#define __GPU_TREE_H

#include "nn.h"
#include "nn_gpu.h"

#define NULL_NODE -1

typedef union gpu_tree_node_0_ {
	long long __val;
	struct {
		int axis;
		float splitval;		 
	} items;
} gpu_tree_node_0;

typedef union gpu_tree_node_1_ {
	double __val[DIM];
	struct {
		float min[DIM];
		float max[DIM];
	} items;
} gpu_tree_node_1;

typedef union gpu_tree_node_2_ {
	long long __val;
	struct {
		int left;
		int right;
	} items;
} gpu_tree_node_2;

typedef struct gpu_tree_node_3_ { 
	int points[MAX_POINTS_IN_CELL];
} gpu_tree_node_3;

typedef struct _gpu_tree {
	gpu_tree_node_0 *nodes0;
	gpu_tree_node_1 *nodes1;
	gpu_tree_node_2 *nodes2;
	gpu_tree_node_3 *nodes3;

	int *stk;

	int nnodes;
	int depth;
} gpu_tree;

typedef struct gpu_point_ {
	int closest;
	float closestDist;
	int numNodesTraversed;
	float coord[DIM];
} gpu_point;

#endif
