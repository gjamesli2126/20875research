/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __GPU_TREE_H__
#define __GPU_TREE_H__

#include "common.h"

extern unsigned int K;
extern ClusterPoint* clusters;
extern unsigned int npoints;
extern DataPoint *points;

gpu_tree * gpu_transform_tree(Node *root);
void gpu_free_tree_host(gpu_tree *h_tree);
static void gpu_alloc_tree_host(gpu_tree * h_tree);
static void gpu_init_tree_properties(gpu_tree *h_tree, Node *root, int depth);
static int gpu_build_tree(Node *root, gpu_tree *h_tree, int *index, int depth, int parent_index);
static gpu_tree* gpu_alloc_tree_dev(gpu_tree *h_tree);
gpu_tree * gpu_copy_to_dev(gpu_tree *h_tree);
void gpu_copy_tree_to_host(gpu_tree *d_tree, gpu_tree *h_tree);
void gpu_free_tree_dev(gpu_tree *d_tree);

#endif
