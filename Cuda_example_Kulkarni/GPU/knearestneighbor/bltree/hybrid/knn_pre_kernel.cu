/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
//
//  BBT_kenrel.cpp
//  BallTree-kNN
//

#include "knn_pre_kernel.h"

#define STACK_NODE node_stack[sp]
//#define POINT cur_node2.point
#define POINT_INDEX cur_node1.idx
//cur_node2.point[s]
#define LEFT cur_node1.left
#define RIGHT cur_node1.right

#define STACK_INIT() \
    sp = 0; 

#define STACK_PUSH() sp = sp + 1; 

#define STACK_POP() sp = sp - 1; 

__global__ void k_nearest_neighbor_pre_search (gpu_tree gpu_tree, int nsearchpoints, datapoint *d_search_points,
											float *d_nearest_distance, int *d_nearest_point_index, int K,
											int* d_matrix, int start, int end, int interval) {
	float search_points[DIM];
	float nearest_distance[8];
	int nearest_point_index[8];

	int pidx;
	int i, j;

	// get cached to registers
	gpu_tree_node_0 cur_node0;
	gpu_tree_node_1 cur_node1;

	int cur_node_index;
	int sp;

	// Get the position of the 1st item
	int node_stack[128];

	float tmpdist;
	int tmpidx;
	int n;
	float t;
	float dist = 0.0;
	float leftPivotDist = 0.0;
	float rightPivotDist = 0.0;

	for (pidx = blockIdx.x * blockDim.x + threadIdx.x; pidx < nsearchpoints; pidx += blockDim.x * gridDim.x) {
		for (j = 0; j < DIM; j++)
			search_points[j] = d_search_points[pidx].coord[j];

		for(i = 0; i < K; i++) {
			nearest_point_index[i] = -1;
			nearest_distance[i] = FLT_MAX;
		}

		// run this for some number of iterations until done...
		STACK_INIT();
        node_stack[0] = 0;
        int d_matrix_index = pidx * interval;
		while(sp >= 0) {
			// get top of stack
			cur_node_index = STACK_NODE;

			if (gpu_tree.nodes1[cur_node_index].depth == SPLICE_DEPTH) {
				d_matrix[d_matrix_index] = gpu_tree.nodes1[cur_node_index].pre_id - 1;
				d_matrix_index ++;
                STACK_POP();
                continue;
			}

			cur_node0 = gpu_tree.nodes0[cur_node_index];
			dist = 0.0;
			for (i = 0; i < DIM; i++) {
				t = (cur_node0.coord[i] - search_points[i]);
				dist +=  t*t;
			}
			dist = sqrt(dist);
			if (dist > -0.000001 && nearest_distance[0] <= (dist - gpu_tree.nodes0[cur_node_index].rad)) {
				STACK_POP();
				continue;
			}

			cur_node1 = gpu_tree.nodes1[cur_node_index];
			if (LEFT == NULL_NODE && RIGHT == NULL_NODE) {
				// update closest point:
				if(dist < nearest_distance[0]) {
					nearest_distance[0] = dist;
    				nearest_point_index[0] = POINT_INDEX;

					// push the value back to maintain sorted order
					for(n = 1; n < K && nearest_distance[n - 1] < nearest_distance[n]; n++) {
						tmpdist = nearest_distance[n];
						tmpidx = nearest_point_index[n];
						nearest_distance[n] = nearest_distance[n-1];
						nearest_point_index[n] = nearest_point_index[n-1];
						nearest_distance[n-1] = tmpdist;
						nearest_point_index[n-1] = tmpidx;
					}
				}
			} else {
				leftPivotDist = 0.0;
				rightPivotDist = 0.0;
				for (i = 0; i < DIM; i++) {
					t = (gpu_tree.nodes0[LEFT].coord[i] - search_points[i]);
					leftPivotDist +=  t*t;
					t = (gpu_tree.nodes0[RIGHT].coord[i] - search_points[i]);
					rightPivotDist +=  t*t;
				}

//				if (gpu_tree.nodes1[cur_node_index].depth < SPLICE_DEPTH) {
					if (leftPivotDist < rightPivotDist) {
						STACK_NODE = RIGHT;
						STACK_PUSH();
						STACK_NODE = LEFT;
						STACK_PUSH();
					} else {
						STACK_NODE = LEFT;
						STACK_PUSH();
						STACK_NODE = RIGHT;
						STACK_PUSH();
					}
//				}
			}
			STACK_POP();
		}

		// Save to global memory
/*		for(i = 0; i < K; i++) {
			d_nearest_point_index[K*pidx+i] = nearest_point_index[i*NUM_THREADS_PER_BLOCK+threadIdx.x];
			d_nearest_distance[K*pidx+i] = nearest_distance[i*NUM_THREADS_PER_BLOCK+threadIdx.x];
		}*/
	}
}

