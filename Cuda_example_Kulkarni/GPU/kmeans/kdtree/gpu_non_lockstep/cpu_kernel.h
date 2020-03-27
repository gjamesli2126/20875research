/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __KMEANS_CPU_KERNEL_H__
#define __KMEANS_CPU_KERNEL_H__
#include "common.h"
#include "functions.h"

extern unsigned int K;
extern ClusterPoint* clusters;
extern unsigned int npoints;
extern DataPoint *points;

float GetDistance(DataPoint* pt, DataPoint* cls);
void KdKmeansCPU (ClusterPoint* clusters, DataPoint* point);
ClusterPoint* NearestNeighbor(Node* root, ClusterPoint* clusters, DataPoint* point);



#endif
// end of __KMEANS_KERNEL_H__
