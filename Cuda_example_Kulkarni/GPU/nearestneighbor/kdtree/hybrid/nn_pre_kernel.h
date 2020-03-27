/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __KNN_BB_PRE_KERNEL_H_
#define __KNN_BB_PRE_KERNEL_H_

#include "nn_data_types.h"
#include "nn_kernel.h"

gpu_tree * pre_transform_tree(KDCell *root);
__global__ void pre_nearest_neighbor_search (kernel_params params, int *d_matrix, int start, int end, int interval);

#endif
