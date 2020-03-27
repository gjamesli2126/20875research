/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __PRE_POINT_CORR_GPU_H_
#define __PRE_POINT_CORR_GPU_H_

#include "../../../common/util_common.h"
#include "pc_data_types.h"

gpu_tree* build_pre_gpu_tree(kd_cell *c_node);
int block_pre_gpu_tree(kd_cell* c_node, gpu_tree* pre_root, int index, int depth);
void free_pre_gpu_tree(gpu_tree *pre_root);
__global__ void pre_compute_correlation(pc_pre_kernel_params params, int start, int end);







#endif
