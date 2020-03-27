/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef common_h
#define common_h

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


#define SAFE_MALLOC(p, n) { p = (typeof(p))malloc(n); if(!p) { fprintf(stderr, "Error: malloc failed to allcate %lu bytes: %s in %s at line %d!\n", n, #p, __FILE__, __LINE__); exit(-1); } }

#define SAFE_CALLOC(p, n, elemsize) { p = (typeof(p))calloc(n, elemsize); if(!p) { fprintf(stderr, "Error: calloc failed to allcate %lu bytes: %s in %s at line %d!\n", n*elemsize, #p, __FILE__, __LINE__); exit(-1); } }

#define CHECK_PTR(p) { if(!p) { fprintf(stderr, "Error: NULL pointer: %s in %s at line %d!\n", #p, __FILE__, __LINE__); exit(-1); }

#define TIME_INIT(timeName) struct timeval timeName ## _start; \
struct timeval timeName ## _end; \
struct timeval timeName ## _temp;

#define TIME_START(timeName) gettimeofday(&(timeName ## _start), NULL);

#define TIME_END(timeName) gettimeofday(&(timeName ## _end), NULL);

#define TIME_RESTART(timeName) timeName ## _temp.tv_sec = timeName ## _end.tv_sec - timeName ## _start.tv_sec; \
timeName ## _temp.tv_usec = timeName ## _end.tv_usec - timeName ## _start.tv_usec; \
TIME_START(timeName); \
timeName ## _start.tv_sec -= timeName ## _temp.tv_sec; \
timeName ## _start.tv_usec -= timeName ## _temp.tv_usec;


#define TIME_ELAPSED(timeName) float timeName ## _elapsed = (timeName ## _end.tv_sec - timeName ## _start.tv_sec); \
timeName ## _elapsed += ((timeName ## _end.tv_usec - timeName ## _start.tv_usec) / (1.0e6)); \
timeName ## _elapsed *= 1000;

#define TIME_ELAPSED_PRINT(timeName, stream) TIME_ELAPSED(timeName) \
fprintf(stream, "@ %s: %2.0f ms\n", #timeName, timeName ## _elapsed);

#ifndef DIM
#define DIM 7
#endif

#ifndef MAX_LABEL
#define MAX_LABEL 8
#endif

using namespace std;

typedef struct datapoint_ {
    float coord[DIM]; //list of dimensions
    int idx;  //its own index
    int label;
    datapoint_() {
        idx = 0;
        for(int i = 0; i < DIM; i ++)
            coord[i] = 0.0;
        label = -1;
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

struct thread_args {
    int tid;
    int lb;
    int ub;
};

#endif /* common_h */
