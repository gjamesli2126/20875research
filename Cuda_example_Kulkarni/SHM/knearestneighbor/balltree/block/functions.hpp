/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef functions_hpp
#define functions_hpp

#include "common.h"

static int label = -1;

extern int sort_flag;
extern int check_flag;
extern int verbose_flag;
extern int warp_flag;
extern int ratio_flag;
#ifdef TRACK_TRAVERSALS
extern long int num_nodes_traversed;
#endif

extern unsigned int npoints;
extern unsigned int nsearchpoints;
extern unsigned int nthreads;

extern unsigned int K;
extern unsigned int max_depth;
extern unsigned int nnodes;

extern node* tree;
extern datapoint *points;
extern datapoint *search_points;
extern neighbor *nearest_neighbor;

void read_input(int argc, char **argv);
node * construct_tree(datapoint *points, int start, int end, datapoint** &datalist, int depth, int id);
float getDistance(datapoint* key, datapoint* curr);
float getDistance2(datapoint* key, datapoint* curr);
pair<float, datapoint*> getRadius(datapoint* target, int start, int end, datapoint** &datalist);
datapoint* getMaxDist(datapoint* target, int start, int end, datapoint** &datalist, vector<float> &distlist);
void sort_search_points(datapoint* points, int start, int end);
static int cmpfn_float(const void *a, const void *b);
void print_result();
void printTree(node* root, int tabs);

#endif /* functions_hpp */
