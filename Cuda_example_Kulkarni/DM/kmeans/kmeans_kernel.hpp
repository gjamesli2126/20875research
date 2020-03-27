/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __KMEANS_KERNEL_H__
#define __KMEANS_KERNEL_H__

#include "common.h"
#include "functions.hpp"
#include "util_common.h"

using namespace boost;
using boost::graph::distributed::mpi_process_group;
extern unsigned int K;
extern ClusterPoint* clusters;
extern unsigned int npoints;
extern DataPoint *points;
#ifdef TRACK_TRAVERSALS
extern long int sum_nodes_traversed;
extern int total_iterations;
#endif

float GetDistance(DataPoint* pt, DataPoint* cls);
void KmeansCPU ();
void* thread_function(void *arg);
void KdKmeansCPU(ClusterPoint* clusters, DataPoint* point, int procRank, int numProcs, mpi::communicator& world);
void NearestNeighbor(Node* root, ClusterPoint* clusters, DataPoint* point, int* clusterId);



#endif
// end of __KMEANS_KERNEL_H__
