/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include "common.h"
#include "functions.hpp"
#include "kmeans_kernel.hpp"
#include "util_common.h"

using namespace boost;
using boost::graph::distributed::mpi_process_group;

int sort_flag = 0;
int check_flag = 0;
int verbose_flag = 0;
int warp_flag = 0;
int ratio_flag = 0;
unsigned int npoints = 0;
unsigned int nthreads = 1;
int DIM = 2;

#ifdef TRACK_TRAVERSALS
long int sum_nodes_traversed = 0;
int total_iterations = 0;
#endif

unsigned int K = 10;

DataPoint *points = NULL;
DataPoint *tmppoints = NULL;

Node* tree = NULL;
unsigned int max_depth = 0;
unsigned int nnodes = 0;
unsigned int avgnodes = 0;

ClusterPoint* clusters = NULL;

TIME_INIT(read_data);
TIME_INIT(build_tree);
TIME_INIT(sort);
TIME_INIT(Kmeans);
TIME_INIT(KdKmeans);

int main(int argc, char** argv) {
	boost::mpi::environment env(argc, argv);
	mpi::communicator world;
	mpi_process_group pg;
    
	int procRank = process_id(pg);
	int numProcs = num_processes(pg);

    TIME_START(read_data);
    read_input(argc, argv, procRank, numProcs);
    TIME_END(read_data);
    srandom(0);
    for (int i = 0; i < K; i ++) {
        clusters[i].num_of_points = 0;
        clusters[i].pt.clusterId = i;
	clusters[i].pt.coord = new float[DIM];
        int j = rand() % npoints;
	tmppoints[j].clusterId = i;
	for(int k=0;k<DIM;k++)
	{
		clusters[i].pt.coord[k] = tmppoints[j].coord[k];
	}
    }  
	free(tmppoints);	 

    TIME_START(KdKmeans);
    KdKmeansCPU(clusters, points, procRank, numProcs, world);
    TIME_END(KdKmeans);
#ifdef TRACK_TRAVERSALS
    long int total_nodes_traversed;
    reduce(world,sum_nodes_traversed,total_nodes_traversed,std::plus<long int>(),0);
    if(procRank == 0)
    {
	printf("The total number of points is %d, and the total number of nodes is %d\n", npoints, avgnodes/total_iterations);
    	printf("Total iterations %d Total number of nodes visited %ld\n",total_iterations, total_nodes_traversed); 
    }
#endif

    return 0;
}
