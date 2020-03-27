/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __KMEANS_COMMON_H__
#define __KMEANS_COMMON_H__

#include "util_common.h"

extern int DIM;

#define MOD(x) (((x) >= 0) ? (x) : (-(x)))

// this structure is defined for easy coord computing in "KdKmeansCPU" (kmeans_kernel.cpp)
// don't use for other purposes.
typedef struct Coord_ {
    float* coord;
    //Coord_(){coord = new float[DIM];}
    //~Coord_(){delete [] coord;}
} Coord;

typedef struct DataPoint_ {
    float* coord; //list of dimensions
    char* label;
    int idx;  //its own index
    int clusterId; // the cluster it belongs to
#ifdef TRACK_TRAVERSALS
    int num_nodes_traversed;
#endif
    DataPoint_() {
	//coord = new float[DIM];
        idx = 0;
        for(int i = 0; i < DIM; i ++)
            coord[i] = 0.0;
	clusterId = -1;
#ifdef TRACK_TRAVERSALS
    num_nodes_traversed=0;
#endif
    }
  //~DataPoint_(){delete [] coord; delete [] label;}
} DataPoint;

typedef struct ClusterPoint_ {
    DataPoint pt; 
    int num_of_points;
} ClusterPoint;

typedef struct Node_ {
#ifdef METRICS
	int numpointsvisited;
#endif
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
#ifdef METRICS
		numpointsvisited=0;
#endif
	}
} Node;

struct thread_args {
    int tid;
    int lb; 
    int ub; 
    Node* root;
    ClusterPoint** nearestclusters;
};

#endif
// end of __KMEANS_COMMON_H__
