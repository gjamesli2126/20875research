/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __BT_GPU_TREE_H_
#define __BT_GPU_TREE_H_

#include "bt_common.h"
#include "bt_functions.h"

#define NULL_NODE -1

typedef struct gpu_tree_node_0_ {
	float coord[DIM];
	float rad;
} gpu_tree_node_0;

typedef struct gpu_tree_node_1_ {
	int left;
	int right;
	int idx;
    int pre_id;
	int depth;
} gpu_tree_node_1;

typedef struct _gpu_tree {
	gpu_tree_node_0 * nodes0;
	gpu_tree_node_1 * nodes1;

	int *stk_node;
	float *stk_axis_dist;

	int nnodes;
	int depth;

} gpu_tree;

extern unsigned int max_depth;
extern unsigned int nnodes;

static int gpu_build_tree(node *root, gpu_tree *h_tree, int *index);
static gpu_tree* gpu_alloc_tree_dev(gpu_tree *h_tree);
gpu_tree * gpu_transform_tree(node *root);
gpu_tree * gpu_copy_to_dev(gpu_tree *h_tree);
void gpu_copy_tree_to_host(gpu_tree *d_tree, gpu_tree *h_tree);
void gpu_free_tree_dev(gpu_tree *d_tree);
void gpu_free_tree_host(gpu_tree *h_tree);
void gpu_print_tree_host(gpu_tree *h_tree);

#endif
