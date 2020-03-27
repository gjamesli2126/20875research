/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __FUNCTIONS_H__
#define __FUNCTIONS_H__

#include "common.h"

extern int sort_flag;
extern int check_flag;
extern int verbose_flag;
extern int warp_flag;
extern int ratio_flag;

extern unsigned int npoints;
extern unsigned int nthreads;

extern unsigned int K;
extern unsigned int max_depth;
extern unsigned int nnodes;

extern Node* tree;
extern DataPoint *points;
extern ClusterPoint* clusters;

static int sort_split;


void read_input(int argc, char **argv);
Node* construct_tree(ClusterPoint *clusters, int start_idx, int end_idx, int depth, Node* parent);
Node * sort_points_by_tree_construction(DataPoint *points, int start_idx, int end_idx, int depth, Node* parent);
void deconstruct_tree(Node* root);
static int cmpfn_float(const void *a, const void *b);
void PrintClusters();

#endif
// end of __FUNCTIONS_H__
