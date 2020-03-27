/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
//
//  functions.hpp
//  BallTree-kNN

#ifndef __BT_FUNCTIONS_H_
#define __BT_FUNCTIONS_H_

#include "knn_common.h"

static int label = -1;

extern int sort_flag;
extern int check_flag;
extern int verbose_flag;
extern int warp_flag;
extern int ratio_flag;

extern unsigned int npoints;
extern unsigned int nsearchpoints;

extern unsigned int K;
extern unsigned int max_depth;
extern unsigned int nnodes;

extern datapoint *points;
extern datapoint *search_points;
extern float* nearest_distance;
extern int* nearest_point_index;


void read_input(int argc, char **argv);
node * construct_tree(datapoint *points, int start, int end, datapoint** &datalist, int depth, int id);
float getDistance(datapoint* key, datapoint* curr);
float getDistance2(datapoint* key, datapoint* curr);
pair<float, datapoint*> getRadius(datapoint* target, int start, int end, datapoint** &datalist);
datapoint* getMaxDist(datapoint* target, int start, int end, datapoint** &datalist, vector<float> &distlist);
void sort_search_points(datapoint* points, int start, int size);
static int cmpfn_float(const void *a, const void *b);
void print_result();
void printTree(node* root, int tabs);

#endif /* functions_hpp */
