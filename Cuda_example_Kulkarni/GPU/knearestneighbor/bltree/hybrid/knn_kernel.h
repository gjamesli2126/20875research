/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
//
//  BBT_kenrel.hpp
//  BallTree-kNN

#ifndef __BT_KERNEL_H_
#define __BT_KERNEL_H_

#include "knn_functions.h"
#include "knn_gpu_tree.h"

extern neighbor *nearest_neighbor;

__global__ void init_kernel(void);
//void k_nearest_neighbor_search(node* node, datapoint* point, int pos);

__global__ void k_nearest_neighbor_search (gpu_tree gpu_tree, int nsearchpoints, datapoint *d_search_points,
											float *d_nearest_distance, int *d_nearest_point_index, int K,
                                            int *index_buffer);

#endif /* BBT_kenrel_hpp */
