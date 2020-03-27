/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __BH_KERNEL_MEM_H
#define __BH_KERNEL_MEM_H

#include <cuda.h>

#include "bh.h"
#include "bh_gpu.h"
#include "bh_gpu_tree.h"

void allocate_gpu_tree_device(bh_gpu_tree * h_root, bh_gpu_tree ** d_root);

void copy_gpu_tree_to_device(bh_gpu_tree * h_root, bh_gpu_tree * d_root);
void copy_gpu_tree_to_host(bh_gpu_tree * h_root, bh_gpu_tree * d_root);
void free_gpu_tree_device(bh_gpu_tree * d_root);

#endif
