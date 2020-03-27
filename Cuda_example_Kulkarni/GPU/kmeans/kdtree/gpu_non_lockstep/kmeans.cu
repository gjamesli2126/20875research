/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include "common.h"
#include "functions.h"
#include "cpu_kernel.h"

int sort_flag = 0;
int check_flag = 0;
int verbose_flag = 0;
int warp_flag = 0;
int ratio_flag = 0;
unsigned int npoints = 0;
unsigned int nthreads = 1;

unsigned int K = 10;

DataPoint *points = NULL;

Node* tree = NULL;
unsigned int max_depth = 0;
unsigned int nnodes = 0;

ClusterPoint* clusters = NULL;

TIME_INIT(read_data);
TIME_INIT(build_tree);
TIME_INIT(sort);
TIME_INIT(Kmeans);
TIME_INIT(KdKmeans);

int main(int argc, char** argv) {
    TIME_START(read_data);
	read_input(argc, argv);
    TIME_END(read_data);
    
    srandom(0);
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

    srandom(0);
    for (int i = 0; i < K; i ++) {
        clusters[i].num_of_points = 0;
        clusters[i].pt.clusterId = i;
        int j = rand() % npoints;
 //       if (points[j].clusterId != -1) {
 //           i --;
 //           continue;
 //       } 
        points[j].clusterId = i;
        for (int k = 0; k < DIM; k ++) {
            clusters[i].pt.coord[k] = points[j].coord[k];
        }      
    }   
//    PrintClusters();

    TIME_START(KdKmeans);
    KdKmeansCPU(clusters, points);
    TIME_END(KdKmeans);
    PrintClusters();

    printf("The total number of points is %d, and the total number of nodes is %d\n", npoints, nnodes);

    TIME_ELAPSED_PRINT(read_data, stdout);
    TIME_ELAPSED_PRINT(Kmeans, stdout);
    TIME_ELAPSED_PRINT(KdKmeans, stdout);
    TIME_ELAPSED_PRINT(build_tree, stdout);


    return 0;
}
