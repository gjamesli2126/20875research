/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __GPU_MEM_H_
#define __GPU_MEM_H_

#include "pc_data_types.h"
#include "pc_kernel.h"

void alloc_tree_dev(gpu_tree *h_root, gpu_tree *d_root);
void alloc_set_dev(gpu_point_set *h_set, gpu_point_set *d_set);
//void alloc_kernel_params_dev(pc_kernel_params *d_params);

void copy_tree_to_dev(gpu_tree *h_root, gpu_tree *d_root);
void copy_set_to_dev(gpu_point_set *h_set, gpu_point_set *d_set);
void copy_tree_to_host(gpu_tree *h_root, gpu_tree *d_root);
void copy_set_to_host(gpu_point_set *h_set, gpu_point_set *d_set);
//void copy_kernel_params_to_dev(pc_kernel_params *h_params, pc_kernel_params *d_params);

void free_tree_dev(gpu_tree *d_root);
void free_set_dev(gpu_point_set *d_set);
//void free_kernel_params_dev(pc_kernel_params *d_params);

#endif
