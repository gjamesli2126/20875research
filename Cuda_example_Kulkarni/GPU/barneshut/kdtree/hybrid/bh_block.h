/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __BH_BLOCK_H
#define __BH_BLOCK_H

#include <cuda.h>
#include "bh_gpu.h"
#include "bh_gpu_tree.h"

bh_gpu_tree * build_gpu_tree(Node * root);
int block_gpu_tree(Node * node, bh_gpu_tree * root, int * index);
void block_tree_info(bh_gpu_tree * gpu_root, Node * root, int depth);

void free_gpu_tree(bh_gpu_tree * root);
void print_gpu_tree(bh_gpu_tree * root);

#endif
