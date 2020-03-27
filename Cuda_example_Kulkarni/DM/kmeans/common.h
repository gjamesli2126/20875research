/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __KMEANS_COMMON_H__
#define __KMEANS_COMMON_H__

#include "util_common.h"
#include<boost/graph/use_mpi.hpp>
#include <boost/graph/distributed/mpi_process_group.hpp>
#include<boost/graph/parallel/algorithm.hpp>
extern int DIM;

#define MOD(x) (((x) >= 0) ? (x) : (-(x)))

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

struct thread_args {
    int tid;
    int lb; 
    int ub; 
    Node* root;
    ClusterPoint** nearestclusters;
};

#endif
// end of __KMEANS_COMMON_H__
