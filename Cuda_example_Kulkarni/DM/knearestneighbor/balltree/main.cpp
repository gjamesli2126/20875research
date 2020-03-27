/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include "common.h"
#include "functions.hpp"
#include "BBT_kernel.hpp"
#include<boost/graph/use_mpi.hpp>
#include <boost/graph/distributed/mpi_process_group.hpp>
#include<boost/graph/parallel/algorithm.hpp>

using namespace boost;
using boost::graph::distributed::mpi_process_group;
int sort_flag = 0;
int check_flag = 0;
int verbose_flag = 0;
int warp_flag = 0;
int ratio_flag = 0;
unsigned int npoints = 0;
unsigned int nsearchpoints = 0;
#ifdef TRACK_TRAVERSALS
long int num_nodes_traversed = 0;
#endif

unsigned int K = 1;

datapoint *points = NULL;
datapoint *search_points = NULL;
neighbor *nearest_neighbor = NULL;

node* tree = NULL;
unsigned int max_depth = 0;
unsigned int nnodes = 0;

TIME_INIT(read_data);
TIME_INIT(build_tree);
TIME_INIT(sort);
TIME_INIT(traversal);

int main(int argc, char * argv[]) {

	boost::mpi::environment env(argc, argv);
	mpi::communicator world;
	mpi_process_group pg;
    
	int procRank = process_id(pg);
	int numProcs = num_processes(pg);
    TIME_START(read_data);
    read_input(argc, argv, procRank, numProcs);
    TIME_END(read_data);
    datapoint** dataList = new datapoint* [npoints];
    for (int i = 0; i < npoints; i ++) {
        dataList[i] = &points[i];
    }
    TIME_START(build_tree);
    tree = construct_tree(points, 0, npoints - 1, dataList, 0, 1);
    TIME_END(build_tree);
    if(procRank == 0)
	printf("The max depth is %d, the nodes number is %d.\n", max_depth, nnodes);
//    printTree(tree, 0);
    
	//sorts only along 0th dimension.
    TIME_START(sort);
    if (sort_flag) {
        sort_search_points(search_points, 0, nsearchpoints);
    }
//    for (int i = 0; i < nsearchpoints; i ++) {
//        printf("%d: point %d, label %d\n", i, search_points[i].idx, search_points[i].label);
//    }
    TIME_END(sort);
    	int numPoints = nsearchpoints;
	int startIndex = procRank * (numPoints/numProcs);
	int endIndex = (procRank == (numProcs - 1))?numPoints:((procRank+1) * (numPoints/numProcs));
	if(numPoints < numProcs)
	{
		if(procRank >= numPoints)
		{
			startIndex = 0;
			endIndex = 0;
		}
		else
		{
			startIndex = procRank;
			endIndex = procRank+1;
		}
	}
    TIME_START(traversal);

    for(int j=0,i = startIndex; i < endIndex; i++,j++) 
    {
        k_nearest_neighbor_search(tree, &search_points[j], j*K);
    }

	synchronize(pg);
    print_result(procRank, numProcs);
    TIME_END(traversal);
 
    float read_data_elapsed_global, build_tree_elapsed_global,sort_elapsed_global,traversal_elapsed_global;
    TIME_ELAPSED(read_data);
    TIME_ELAPSED(build_tree);
    TIME_ELAPSED(sort);
    TIME_ELAPSED(traversal);
    reduce(world, traversal_elapsed, traversal_elapsed_global, boost::parallel::maximum<float>(),0);
#ifdef TRACK_TRAVERSALS
    long int num_nodes_traversed_global;
    reduce(world, num_nodes_traversed, num_nodes_traversed_global, std::plus<long int>(),0);
    if(procRank == 0)
	printf("Total nodes traversed %ld\n", num_nodes_traversed_global);
#endif
    if(procRank == 0)
	printf("Traversal time %f seconds\n", traversal_elapsed_global/1000);

    /*TIME_ELAPSED_PRINT(read_data, stdout);
    TIME_ELAPSED_PRINT(build_tree, stdout);
    TIME_ELAPSED_PRINT(sort, stdout);
    TIME_ELAPSED_PRINT(traversal, stdout);*/
    
    delete [] points;
    delete [] search_points;
    delete [] dataList;
    
    return 0;
}

