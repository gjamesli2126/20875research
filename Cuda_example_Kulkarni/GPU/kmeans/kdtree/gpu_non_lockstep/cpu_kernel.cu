/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include "cpu_kernel.h"
#include "gpu_kernel.h"
#include "gpu_tree.h"
#include "common.h"
TIME_INIT(kernel);

float GetDistance(DataPoint* pt, DataPoint* cls) {
    float dist = 0.0f;
    for (int i = 0; i < DIM; i ++) {
        dist += (pt->coord[i] - cls->coord[i]) * (pt->coord[i] - cls->coord[i]);
    }
    return sqrt(dist);
}

void KdKmeansCPU(ClusterPoint* clusters, DataPoint* points) {
    int i = -1, inCluster = -1;
    bool changed = false;
    Coord* tempCoords = NULL;
    int* tempClustedId = NULL;
    ClusterPoint* nearestCluster = NULL;
    Node* root = NULL;

    tempCoords = (Coord*) malloc (sizeof(Coord) * K);
    tempClustedId = (int*) malloc (sizeof(int) * npoints);

    Node* root_for_points = sort_points_by_tree_construction(points, 0, npoints-1, 0, NULL);

    do {
        memset(tempCoords, 0, sizeof(Coord) * K);
//        memset(tempClustedId, 0, sizeof(int) * npoints);
        for(int i = 0; i < K; i ++) {
            clusters[i].num_of_points = 0;
        }
        for(int i = 0; i < npoints; i ++) {
            tempClustedId[i] = points[i].clusterId;
        }

        for(int i = 0; i < K; i ++) {
            clusters[i].pt.clusterId = i;
        }
        root = construct_tree(clusters, 0, K - 1, 0, NULL);

//        1. construct the GPU tree
        gpu_tree *h_tree = gpu_transform_tree(root);
//        2. copy the tree to GPU
        gpu_tree *d_tree = gpu_copy_to_dev(h_tree);
//        3. copy the points to GPU
        DataPoint *d_points;
        CUDA_SAFE_CALL(cudaMalloc(&d_points, sizeof(DataPoint)*npoints));
        CUDA_SAFE_CALL(cudaMemcpy(d_points, points, sizeof(DataPoint)*npoints, cudaMemcpyHostToDevice));
//        4. do GPU computation

        init_kernel<<<1,1>>>();
        dim3 blocks(NUM_OF_BLOCKS);
        dim3 tpb(NUM_OF_THREADS_PER_BLOCK);
        TIME_RESTART(kernel);
        nearest_cluster<<<blocks, tpb>>>(*d_tree, d_points, npoints, K);
        cudaError_t err = cudaThreadSynchronize();
        if(err != cudaSuccess) {
            fprintf(stderr,"Kernel failed with error: %s\n", cudaGetErrorString(err));
            exit(1);
        }
        TIME_END(kernel);
//        5. copy back the result
        CUDA_SAFE_CALL(cudaMemcpy(points, d_points, sizeof(DataPoint)*npoints, cudaMemcpyDeviceToHost));
//        6. clean up the space
        gpu_free_tree_host(h_tree);
        gpu_free_tree_dev(d_tree);
        CUDA_SAFE_CALL(cudaFree(d_points));

        for(int i = 0; i < npoints; i ++) {
            inCluster = points[i].clusterId;
            clusters[inCluster].num_of_points ++;
            for(int j = 0; j < DIM; j ++) {
                tempCoords[inCluster].coord[j] += points[i].coord[j];
            }
        }
        deconstruct_tree(root);

        // Compute new centroid
        for (int i = 0; i < K; i ++) {
            assert(clusters[i].num_of_points != 0);
            for (int j = 0; j < DIM; j ++) {
                clusters[i].pt.coord[j] = tempCoords[i].coord[j] / clusters[i].num_of_points;
            }
        }

        // check if anything has changed
        changed = false;
        for (int i = 0; i < npoints; i ++) {
            if (points[i].clusterId != tempClustedId[i]) {
                changed = true;
                break;
            }
        }

    } while (changed);
    TIME_ELAPSED_PRINT(kernel, stdout);
}


