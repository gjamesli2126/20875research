/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __KMEANS_COMMON_H__
#define __KMEANS_COMMON_H__

#include "../../../common/util_common.h"
#ifndef DIM
#define DIM 7
#endif

#define MOD(x) (((x) >= 0) ? (x) : (-(x)))

// this structure is defined for easy coord computing in "KdKmeansCPU" (kmeans_kernel.cpp)
// don't use for other purposes.
typedef struct Coord_ {
    float coord[DIM];
} Coord;

typedef struct DataPoint_ {
    float coord[DIM]; //list of dimensions
    int idx;  //its own index
    int label;
	int clusterId; // the cluster it belongs to
#ifdef TRACK_TRAVERSALS
    long numNodesTraversed;
#endif
    DataPoint_() {
        idx = 0;
        for(int i = 0; i < DIM; i ++)
            coord[i] = 0.0;
        label = -1;
		clusterId = -1;
#ifdef TRACK_TRAVERSALS
        numNodesTraversed = 0;
#endif
    }
} DataPoint;

typedef struct ClusterPoint_ {
    DataPoint pt; 
    int num_of_points;
} ClusterPoint;

typedef struct Node_ {
    int axis;
//    int node_index;
    ClusterPoint *pivot;
    struct Node_ *left;
    struct Node_ *right;
    struct Node_ *parent;
    int clusterId;
    int depth;

	Node_() {
		axis = -1;
		pivot = NULL;
		left = NULL;
		right = NULL;
	}
} Node;

typedef struct gpu_tree_node_0_ {
    int axis;
    int clusterId;
    int depth;
    int parent;
} gpu_tree_node_0;

typedef struct gpu_tree_node_1_ {
    float coord[DIM];
} gpu_tree_node_1;

typedef struct gpu_tree_node_2_ {
    int left;
    int right;
} gpu_tree_node_2;

typedef struct gpu_tree_node_3_ { 
//    int points[MAX_POINTS_IN_CELL];
} gpu_tree_node_3;

typedef struct _gpu_tree {
    gpu_tree_node_0 *nodes0;
    gpu_tree_node_1 *nodes1;
    gpu_tree_node_2 *nodes2;
    gpu_tree_node_3 *nodes3;

    bool* visited;
    int nnodes;
    int depth;
} gpu_tree;

#endif
// end of __KMEANS_COMMON_H__
