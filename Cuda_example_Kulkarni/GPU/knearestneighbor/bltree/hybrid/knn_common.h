/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
//
//  common.h
//  BallTree-kNN

#ifndef __BT_COMMON_H_
#define __BT_COMMON_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdbool.h>
#include <vector>
#include <ctype.h>
#include <assert.h>
#include <sys/time.h>
#include <string>
#include <iostream>
#include <time.h>
#include<queue>
#include<cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include<limits>
#include "../../../common/util_common.h"

#ifndef DIM
#define DIM 7
#endif

#ifndef MAX_LABEL
#define MAX_LABEL DIM+1
#endif

using namespace std;

typedef struct datapoint_ {
    float coord[DIM]; //list of dimensions
    int idx;  //its own index
    int label;
	
#ifdef TRACK_TRAVERSALS
	int numNodesTraversed;
#endif
    datapoint_() {
        idx = 0;
        for(int i = 0; i < DIM; i ++)
            coord[i] = 0.0;
        label = -1;
#ifdef TRACK_TRAVERSALS
		numNodesTraversed = 0;
#endif
    }
} datapoint;

typedef struct node_ {
    float rad;
    datapoint *pivot;
    
    struct node_ *left;
    struct node_ *right;
    
    int depth;
    int pre_id;
    
    node_(){
        this->pivot = NULL;
        this->left = NULL;
        this->right = NULL;
        this->rad = 0;
        this->depth = 0;
        this->pre_id = 0;
    }
} node;

typedef struct neighbor_ {
    float dist;
    datapoint* point;
    neighbor_() {
        dist = 0.0;
        point = NULL;
    }
} neighbor;

#define DOWN 0
#define UP 1

#ifndef NUM_OF_WARPS_PER_BLOCK
#define NUM_OF_WARPS_PER_BLOCK          6
#endif 

#ifndef NUM_OF_BLOCKS
#define NUM_OF_BLOCKS                   (1024)
#endif 

#define NUM_OF_THREADS_PER_WARP         32
#define NUM_OF_THREADS_PER_BLOCK        NUM_OF_WARPS_PER_BLOCK * NUM_OF_THREADS_PER_WARP

#define WARP_IDX (threadIdx.x >> 5)
#define GLOBAL_WARP_IDX (WARP_IDX + (blockIdx.x*NUM_WARPS_PER_BLOCK))
#define THREAD_IDX_IN_WARP (threadIdx.x - (WARP_SIZE * WARP_IDX))
#define IS_FIRST_THREAD_IN_WARP (threadIdx.x == (WARP_IDX * WARP_SIZE)

#endif /* common_h */
