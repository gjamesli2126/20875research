/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __POINT_CORR_FUNCTIONS_H_
#define __POINT_CORR_FUNCTIONS_H_

#include "pc_data_types.h"

extern int npoints; // number of input points
extern kd_cell **points; // input points
extern kd_cell *root; // root of the tree
extern int sort_flag;
extern int verbose_flag;
extern int check_flag;
extern int ratio_flag;
extern int warp_flag;
extern int nthreads;
extern unsigned long corr_sum;
extern unsigned int sum_of_nodes;
extern gpu_tree *h_root; // root of the host GPU tree
extern gpu_tree d_root;
extern gpu_tree *h_pre_root; // root of the pre GPU tree
extern gpu_tree d_pre_root;
extern bool *h_correlation_matrix;
extern bool *d_correlation_matrix;
extern unsigned int COLS;
extern unsigned int ROWS;
extern int sort_split; // axis component compared
extern int sortidx;

void read_input(int argc, char *argv[]);
kd_cell * build_tree(kd_cell ** points, int split, int lb, int ub, int index);
int kdnode_cmp(const void *a, const void *b);
void free_tree(kd_cell *root);
static inline float distance_axis(kd_cell *a, kd_cell *b, int axis);
static inline float distance(kd_cell *a, kd_cell *b);
bool can_correlate(kd_cell *point, kd_cell *cell);





#endif
