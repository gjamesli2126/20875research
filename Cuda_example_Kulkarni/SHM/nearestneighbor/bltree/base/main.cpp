/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include "common.h"
#include "functions.hpp"
#include "BBT_kernel.hpp"

int sort_flag = 0;
int check_flag = 0;
int verbose_flag = 0;
int warp_flag = 0;
int ratio_flag = 0;
unsigned int npoints = 0;
unsigned int nsearchpoints = 0;
unsigned int nthreads = 1;

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

void* thread_function(void *arg) {
    struct thread_args *args = (struct thread_args*)arg;
    for(int i = args->lb; i < args->ub; i++) {
        k_nearest_neighbor_search(tree, &search_points[i], i*K);
    }
    return NULL;
}

int main(int argc, char * argv[]) {
    
    TIME_START(read_data);
    read_input(argc, argv);
    TIME_END(read_data);
    datapoint** dataList = new datapoint* [npoints];
    for (int i = 0; i < npoints; i ++) {
        dataList[i] = &points[i];
    }
    TIME_START(build_tree);
    tree = construct_tree(points, 0, npoints - 1, dataList, 0, 1);
    TIME_END(build_tree);
    printf("The max depth is %d, the nodes number is %d.\n", max_depth, nnodes);
//    printTree(tree, 0);
    
    TIME_START(sort);
    if (sort_flag) {
        sort_search_points(search_points, 0, nsearchpoints);
    }
//    for (int i = 0; i < nsearchpoints; i ++) {
//        printf("%d: point %d, label %d\n", i, search_points[i].idx, search_points[i].label);
//    }
    TIME_END(sort);
    
    struct thread_args *args;
    pthread_t *threads;
    SAFE_MALLOC(args, sizeof(struct thread_args)*nthreads);
    SAFE_MALLOC(threads, sizeof(pthread_t)*nthreads);
    
    // Assign points to threads
    int start = 0;
    for(int i = 0; i < nthreads; i++) {
        int num = (nsearchpoints - start) / (nthreads - i);
        args[i].tid = i;
        args[i].lb = start;
        args[i].ub = start + num;
        start += num;
        
        //printf("%d %d\n", args[j].lb, args[j].ub);
    }
    
    TIME_START(traversal);
    for(int i = 0; i < nthreads; i++) {
        if(pthread_create(&threads[i], NULL, thread_function, &args[i]) != 0) {
            fprintf(stderr, "Could not create thread %d\n", i);
            exit(1);
        }
    }
    
    for(int i = 0; i < nthreads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    free(args);
    free(threads);
    TIME_END(traversal);
    print_result();
    
    TIME_ELAPSED_PRINT(read_data, stdout);
    TIME_ELAPSED_PRINT(build_tree, stdout);
    TIME_ELAPSED_PRINT(sort, stdout);
    TIME_ELAPSED_PRINT(traversal, stdout);
    
    delete [] points;
    delete [] search_points;
    delete [] dataList;
    
    return 0;
}

