/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __POINT_CORR_GPU_H_
#define __POINT_CORR_GPU_H_

#include "../../../common/util_common.h"
#include "pc_data_types.h"

gpu_tree * build_gpu_tree(kd_cell * c_root);
int block_gpu_tree(kd_cell * c_node, gpu_tree * h_root, int index, int depth);
void block_tree_info(gpu_tree * h_root, kd_cell * c_root, int depth);
void free_gpu_tree(gpu_tree * h_root);

__global__ void init_kernel(void);
__global__ void compute_correlation(pc_kernel_params params);

#endif
