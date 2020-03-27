/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __KNN_BB_MEM_H_
#define __KNN_BB_MEM_H_

#include <cuda.h>
#include "nn_data_types.h"
#include "nn_kernel.h"
#include "../../../common/util_common.h"

gpu_tree* alloc_tree_dev(gpu_tree *h_tree);
gpu_tree* copy_tree_to_dev(gpu_tree *h_tree);
void free_tree_dev(gpu_tree *d_tree);

gpu_point * gpu_transform_points(Point *points, unsigned int npoints);
void gpu_free_points_host(gpu_point *h_points);
void gpu_free_points_dev(gpu_point *d_points);
gpu_point *gpu_copy_points_to_dev(gpu_point *h_points, unsigned int npoints);
//void gpu_copy_points_to_host(gpu_point *d_points, gpu_point *h_points, SpliceNode *sn, unsigned int npoints);
void gpu_copy_points_to_host(gpu_point *d_points, gpu_point *h_points, Point *points, unsigned int npoints);

void alloc_set_dev(gpu_point_set *h_set, gpu_point_set *d_set);
void copy_set_to_dev(gpu_point_set *h_set, gpu_point_set *d_set);
void copy_set_to_host(gpu_point_set *h_set, gpu_point_set *d_set);
void free_set_dev(gpu_point_set *d_set);

#endif
