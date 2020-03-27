/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __BH_GPU_PARAMS_H
#define __BH_GPU_PARAMS_H

#include "bh.h"
#include "ptrtab.h"

typedef struct _bh_kernel_params {
	bh_oct_tree_gpu root;
	//bh_kernel_stacks stacks;
	
	int nbodies; 
	float itolsq;
	int step;
	float dthf;
	float epssq;
  float size;
	
	bh_oct_tree_node **h_points_sorted;
	bh_oct_tree_node **d_points_sorted; // index values of points_sorted array

	hash_table_t ptrtab_points_sorted; // pointer table for points_sorted

	int *stk_nodes;
	float *stk_dsq;

} bh_kernel_params;

#endif
