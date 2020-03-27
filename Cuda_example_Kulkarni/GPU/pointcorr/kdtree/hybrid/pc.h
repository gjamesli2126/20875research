/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __POINT_CORR_H_
#define __POINT_CORR_H_

#include "../../../common/util_common.h"
#include "pc_data_types.h"
#include "pc_pre_kernel.h"
#include "pc_kernel.h"
#include "pc_mem.h"
#include "pc_functions.h"
#include <list>

void read_input(int argc, char *argv[]);
kd_cell * build_tree(kd_cell ** points, int split, int lb, int ui, int index);
int kdnode_cmp(const void *a, const void *b);
void print_tree(kd_cell * root, int depth);
void free_tree(kd_cell *root);

void find_correlation(int start, int end);
bool can_correlate(kd_cell *point, kd_cell *cell);

static inline float distance_axis(kd_cell *a, kd_cell *b, int axis);
static inline float distance(kd_cell *a, kd_cell *b);

#endif
