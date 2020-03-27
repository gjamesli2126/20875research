/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __KNN_BB_DATA_TYPES_H_
#define __KNN_BB_DATA_TYPES_H_

#include "../../../common/util_common.h"

const int MAX_POINTS_IN_CELL = 1;

typedef struct Point 
{
	int label;
	float coord[DIM];
	int closest;
	float closest_dist;
	int id;
#ifdef TRACK_TRAVERSALS
	int num_nodes_traversed;
#endif
} Point;

typedef struct KDCell
{
	int axis;
	float splitval;
	int depth;
	int pre_id;
	float min[DIM];
	float max[DIM];
	int points[MAX_POINTS_IN_CELL];
	struct KDCell *left;
	struct KDCell *right;
	int id;
} KDCell;

#define NULL_NODE -1

typedef union gpu_tree_node_0_ 
{
	long long __val;
	struct 
	{
		int axis;
		float splitval;	
		int depth;
		int pre_id;
	} items;
} gpu_tree_node_0;

typedef union gpu_tree_node_1_ 
{
	double __val[DIM];
	struct {
		float min;
		float max;
	} items[DIM];
} gpu_tree_node_1;

typedef union gpu_tree_node_2_ 
{
	long long __val;
	struct 
	{
		int left;
		int right;
	} items;
} gpu_tree_node_2;

typedef struct gpu_tree_node_3_ 
{ 
	long long __val;
	struct 
	{
		int points[MAX_POINTS_IN_CELL];
		int my_index;
	} items;
} gpu_tree_node_3;

typedef struct _gpu_tree
{
	gpu_tree_node_0 *nodes0;
	gpu_tree_node_1 *nodes1;
	gpu_tree_node_2 *nodes2;
	gpu_tree_node_3 *nodes3;

	int *stk;
	int max_nnodes;
	int nnodes;
	int depth;
} gpu_tree;

typedef struct gpu_point_ 
{
	int closest;
#ifdef TRACK_TRAVERSALS
	int num_nodes_traversed;
#endif
	int id;
	float closest_dist;
	float coord[DIM];
} gpu_point;

typedef struct gpu_point_set_
{
	int npoints;
	gpu_point *points;
} gpu_point_set;

struct thread_args 
{
	int tid;
	int lb;
	int ub;	
};

struct kernel_params
{
    gpu_tree d_tree;
    gpu_point *d_training_points;
    int n_training_points;
    gpu_point *d_search_points;
    int n_search_points;
    int *d_array_points;
    int num_of_leafs;
    int n_root_index;
};

#endif






