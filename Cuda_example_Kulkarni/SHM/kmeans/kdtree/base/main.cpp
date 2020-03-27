/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/

#include "common.h"
#include "functions.hpp"
#include "kmeans_kernel.hpp"
#include "util_common.h"
#include<math.h>

#ifdef PAPI
int eventSet=PAPI_NULL;
long long values[4];
void handle_error(int retval)
{
	printf("PAPI error %d: %s\n", retval, PAPI_strerror(retval));
	exit(1);
}
#endif


unsigned int npoints = 0;
unsigned int nthreads = 1;
int DIM = 2;
#ifdef METRICS
int splice_depth=0;
std::vector<Node*> subtrees;
std::vector<subtreeStats> subtreeStatsList;
#endif


#ifdef TRACK_TRAVERSALS
long int sum_nodes_traversed = 0;
int total_iterations = 0;
#endif

unsigned int K = 10;

DataPoint *points = NULL;

Node* tree = NULL;
unsigned int nnodes = 0;
unsigned int avgnodes = 0;
unsigned int max_depth = 0;

ClusterPoint* clusters = NULL;

TIME_INIT(read_data);
TIME_INIT(build_tree);
TIME_INIT(sort);
TIME_INIT(Kmeans);
TIME_INIT(KdKmeans);

int main(int argc, char** argv) {
#ifdef PAPI
	int retval;

	retval = PAPI_library_init(PAPI_VER_CURRENT);
	if(retval != PAPI_VER_CURRENT)
		handle_error(retval);
	
	retval = PAPI_multiplex_init();
	if (retval != PAPI_OK) 
		handle_error(retval);
	retval = PAPI_create_eventset(&eventSet);
	if (retval != PAPI_OK) 
		handle_error(retval);
 
	// Add Total L2Cache Misses 
	retval = PAPI_add_event(eventSet, PAPI_L3_TCM);
	if (retval != PAPI_OK) 
		handle_error(retval);
	
	// Total L1 cache accesses = total memory accesses. Needed for computing L2 miss rate. On Qstruct, there are 2 layers of cache. 
	retval = PAPI_add_event(eventSet, PAPI_L3_TCA);
	if (retval != PAPI_OK) 
		handle_error(retval);

	retval = PAPI_set_multiplex(eventSet);
	if (retval != PAPI_OK) 
		handle_error(retval);

	// TOTAL cycles 
	retval = PAPI_add_event(eventSet, PAPI_TOT_CYC);
	if (retval != PAPI_OK) 
		handle_error(retval);
	
	// TOTAL instructions 
	retval = PAPI_add_event(eventSet, PAPI_TOT_INS);
	if (retval != PAPI_OK) 
		handle_error(retval);

#endif

    TIME_START(read_data);
	read_input(argc, argv);
    TIME_END(read_data);
    
    /*srandom(0);
    for (int i = 0; i < K; i ++) {
        clusters[i].num_of_points = 0;
        clusters[i].pt.clusterId = i;
        int j = rand() % npoints;
        if (points[j].clusterId != -1) {
            i --;
            continue;
        } 
        points[j].clusterId = i;
        for (int k = 0; k < DIM; k ++) {
            clusters[i].pt.coord[k] = points[j].coord[k];
        }      
    }	
//    PrintClusters();   
    printf("************************\n\n");
 
    TIME_START(Kmeans);
    KmeansCPU(); 
    TIME_END(Kmeans);
#ifdef DEBUG
    FILE *delme_brute=fopen("delme_brute","w");
    PrintClusters(delme_brute);
    fclose(delme_brute);	
#endif

    TIME_START(build_tree);
    tree = construct_tree(clusters, 0, K - 1, 0, NULL);
    deconstruct_tree(tree);
    TIME_END(build_tree);*/

    srandom(0);
    for (int i = 0; i < K; i ++) {
        clusters[i].num_of_points = 0;
        clusters[i].pt.clusterId = i;
	clusters[i].pt.coord = new float[DIM];
        int j = rand() % npoints;
        points[j].clusterId = i;
        for (int k = 0; k < DIM; k ++) {
            clusters[i].pt.coord[k] = points[j].coord[k];
        }      
    }   
//    PrintClusters();

#ifdef METRICS
	int max_depth = log2(K);
	if(max_depth % 2)
		splice_depth = max_depth/2+1;
	else
		splice_depth = max_depth/2;	
	printf("splice depth %d\n",splice_depth);
#endif
    TIME_START(KdKmeans);
    KdKmeansCPU(clusters, points);
    TIME_END(KdKmeans);
    printf("The total number of points is %d, and the total number of nodes is %d max depth %d\n", npoints, avgnodes/total_iterations, max_depth);
#ifdef TRACK_TRAVERSALS
	printf("Total iterations %d Total number of nodes visited %ld\n",total_iterations, sum_nodes_traversed); 
#endif
#ifdef DEBUG
    FILE *delme_kdtree=fopen("delme_kdtree","w");
    PrintClusters(delme_kdtree);
    fclose(delme_kdtree);	
#endif
#ifdef METRICS
	std::vector<subtreeStats>::iterator iter=subtreeStatsList.begin();
	for(;iter!=subtreeStatsList.end();iter++)
	{
		printf("vertices %d footprint %d\n",iter->numnodes,iter->footprint);
	}
#endif
    /*TIME_ELAPSED_PRINT(read_data, stdout);
    TIME_ELAPSED_PRINT(Kmeans, stdout);
    TIME_ELAPSED_PRINT(KdKmeans, stdout);
    TIME_ELAPSED_PRINT(build_tree, stdout);*/


    return 0;
}
