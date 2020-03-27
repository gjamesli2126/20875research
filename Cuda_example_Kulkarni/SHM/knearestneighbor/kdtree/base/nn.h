/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef NN_H_
#define NN_H_

#include "util_common.h"
#ifndef DIM
#define DIM 7
#endif

typedef struct node_ {
	int axis;
	int point_index;
	float point[DIM];

	struct node_ *left;
	struct node_ *right;

} node;

struct thread_args {
	int tid;
	int lb;
	int ub;	
};


#define distance_axis(a, b, axis) ((a[axis]-b[axis])*(a[axis]-b[axis]))

void read_input(int argc, char **argv);
void read_point(node *p, FILE *in, int index, int random);

node *construct_tree(node *points, int start_idx, int end_idx, int depth);
void sort_search_points(float *points, int start_idx, int end_idx, int depth);
void quick_sort(float *points, int axis, int lb, int ub);
void print_tree(node * root);

void nearest_neighbor_search(node *point, node *current_node, int pidx, float axis_dist);
void nearest_neighbor_search_brute(node *point, int pidx);

#endif /* NN_H_ */
