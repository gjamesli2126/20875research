/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __KNN_BB_FUNCTIONS_H_
#define __KNN_BB_FUNCTIONS_H_

#include "nn_data_types.h"
#include "../../../common/util_common.h"

extern Point *training_points;
extern KDCell *root;
extern Point *search_points;

extern int sort_flag;
extern int verbose_flag;
extern int check_flag;
extern int ratio_flag;
extern int warp_flag;
extern int nthreads;

extern int npoints;
extern int nsearchpoints;

void read_input(int argc, char **argv);
void read_point(FILE *in, Point *p);

int can_correlate(Point * point, KDCell * cell, float rad);
//void update_closest(Point *point, Point *candidate);
void update_closest(Point *point, int candidate_index);

static inline float distance_axis(Point *a, Point *b, int axis);
static inline float distance(Point *a, Point *b);
KDCell* construct_tree(Point *points, int start_idx, int end_idx, int depth, int index);
void sort_points(Point *points, int start_idx, int end_idx, int depth);
int compare_point(const void *a, const void *b);
void free_tree(KDCell *n);
Point* alloc_points(int n);
KDCell* alloc_kdcell();

#endif
