/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __BH_OCT_TREE_GPU_H
#define __BH_OCT_TREE_GPU_H

#include "bh.h"

typedef struct gpu_node0_ {
	Vec cofm;
	int point_id;
	bool leafNode;
} gpu_node0;

typedef struct gpu_node1_ {
	float mass;
	float ropen;
} gpu_node1;

typedef struct gpu_node2_ {
	int left;
	int right;
} gpu_node2;

typedef struct gpu_node3_ {
//	bh_oct_tree_node *cpu_addr;
	int pre_id;
	int depth;
	
	#ifdef TRACK_TRAVERSALS
	int nodes_accessed;
	#endif
} gpu_node3;

typedef union {
	long long __val;
	struct {
		int index;
		float dsq;
	} items;
} stack_item;

typedef struct _bh_gpu_tree_ {
	gpu_node0 *nodes0;
	gpu_node1 *nodes1;
	gpu_node2 *nodes2;
	gpu_node3 *nodes3;
	unsigned int nnodes;
	unsigned int depth;	
		
	int *stk_nodes;
	float *stk_dsq;
} bh_gpu_tree;

typedef struct gpu_point_ {
	float mass;
	Vec cofm; /* center of mass */
	Vec vel; /* current velocity */
	Vec acc; /* current acceleration */
} gpu_point;
#endif
