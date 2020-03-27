/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __BH_BLOCK_H
#define __BH_BLOCK_H

#include <cuda.h>
#include "bh_gpu.h"
#include "bh_gpu_tree.h"
#include "bh_block_info.h"

bh_oct_tree_gpu * build_gpu_tree(bh_oct_tree_node * root);
int block_gpu_tree(bh_oct_tree_node * node, bh_oct_tree_gpu * root, int * index, int max_block_size, int depth);
void block_tree_info(bh_oct_tree_gpu * gpu_root, bh_oct_tree_node * root, int depth);

void free_gpu_tree(bh_oct_tree_gpu * root);
void print_gpu_tree(bh_oct_tree_gpu * root);

#endif
