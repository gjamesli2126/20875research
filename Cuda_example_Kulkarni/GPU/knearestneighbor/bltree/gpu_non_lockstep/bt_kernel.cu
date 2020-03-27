/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
//
//  BBT_kenrel.cpp
//  BallTree-kNN

#include "bt_kernel.h"

__global__ void init_kernel(void) {

}

__global__ void k_nearest_neighbor_search (gpu_tree gpu_tree, int nsearchpoints, datapoint *d_search_points,
											float *d_nearest_distance, int *d_nearest_point_index, int K) {
	extern __shared__ int smem[];
	float *search_points;
	float *nearest_distance;
	int *nearest_point_index;

	int pidx;
	int i, j;

	// get cached to registers
	gpu_tree_node_0 cur_node0;
	gpu_tree_node_1 cur_node1;

	int cur_node_index;
	int sp;

	// Get the position of the 1st item
	int *stk_node;
	int stk_node_top;

	float tmpdist;
	int tmpidx;
	int n;
	float t;
	float dist = 0.0;
	float leftPivotDist = 0.0;
	float rightPivotDist = 0.0;

	// setup shared memory pointers:
	// It seems like these all need to be alligned to 32? byte boundry?
	unsigned int off=0;
	search_points = (float*)(&smem[off]);

	off = NUM_THREADS_PER_BLOCK*DIM;
	nearest_point_index = (int*)(&smem[off]);

	off = off + NUM_THREADS_PER_BLOCK*K;
	nearest_distance = (float*)(&smem[off]);
	for (pidx = blockIdx.x * blockDim.x + threadIdx.x; pidx < nsearchpoints; pidx += blockDim.x * gridDim.x) {
		for (j = 0; j < DIM; j++)
			search_points[NUM_THREADS_PER_BLOCK*j+threadIdx.x] = d_search_points[pidx].coord[j];

		for(i = 0; i < K; i++) {
			nearest_point_index[i*NUM_THREADS_PER_BLOCK+threadIdx.x] = -1;
			nearest_distance[i*NUM_THREADS_PER_BLOCK+threadIdx.x] = FLT_MAX;
		}

		// run this for some number of iterations until done...
		STACK_INIT();
		while(sp >= 0) {
			// get top of stack
			cur_node_index = STACK_NODE;

			cur_node0 = gpu_tree.nodes0[cur_node_index];
			dist = 0.0;
			for (i = 0; i < DIM; i++) {
				t = (cur_node0.coord[i] - search_points[NUM_THREADS_PER_BLOCK*i + threadIdx.x]);
				dist +=  t*t;
			}
			dist = sqrt(dist);
			if (dist > -0.000001 && nearest_distance[threadIdx.x] <= (dist - gpu_tree.nodes0[cur_node_index].rad)) {
				STACK_POP();
				continue;
			}

			cur_node1 = gpu_tree.nodes1[cur_node_index];
			if (LEFT == NULL_NODE && RIGHT == NULL_NODE) {
				// update closest point:
				if(dist < nearest_distance[threadIdx.x]) {
					nearest_distance[threadIdx.x] = dist;
					nearest_point_index[threadIdx.x] = POINT_INDEX;

					// push the value back to maintain sorted order
					for(n = 1; n < K && nearest_distance[(n - 1)*NUM_THREADS_PER_BLOCK+threadIdx.x] < nearest_distance[(n*NUM_THREADS_PER_BLOCK)+threadIdx.x]; n++) {
						tmpdist = nearest_distance[n*NUM_THREADS_PER_BLOCK+threadIdx.x];
						tmpidx = nearest_point_index[n*NUM_THREADS_PER_BLOCK+threadIdx.x];
						nearest_distance[n*NUM_THREADS_PER_BLOCK+threadIdx.x] = nearest_distance[(n-1)*NUM_THREADS_PER_BLOCK+threadIdx.x];
						nearest_point_index[n*NUM_THREADS_PER_BLOCK+threadIdx.x] = nearest_point_index[(n-1)*NUM_THREADS_PER_BLOCK+threadIdx.x];
						nearest_distance[(n-1)*NUM_THREADS_PER_BLOCK+threadIdx.x] = tmpdist;
						nearest_point_index[(n-1)*NUM_THREADS_PER_BLOCK+threadIdx.x] = tmpidx;
					}
				}
//				STACK_POP();
			} else {
				leftPivotDist = 0.0;
				rightPivotDist = 0.0;
				for (i = 0; i < DIM; i++) {
					t = (gpu_tree.nodes0[LEFT].coord[i] - search_points[NUM_THREADS_PER_BLOCK*i + threadIdx.x]);
					leftPivotDist +=  t*t;
				}
				for (i = 0; i < DIM; i++) {
					t = (gpu_tree.nodes0[RIGHT].coord[i] - search_points[NUM_THREADS_PER_BLOCK*i + threadIdx.x]);
					rightPivotDist +=  t*t;
				}

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
			}
            STACK_POP();
		}

		// Save to global memory
		for(i = 0; i < K; i++) {
			d_nearest_point_index[K*pidx+i] = nearest_point_index[i*NUM_THREADS_PER_BLOCK+threadIdx.x];
			d_nearest_distance[K*pidx+i] = nearest_distance[i*NUM_THREADS_PER_BLOCK+threadIdx.x];
		}
	}
}


/*void k_nearest_neighbor_search(node* node, datapoint* point, int pos) {
    float dist = getDistance(point, node->pivot);
    neighbor* pair_list = &nearest_neighbor[pos];
    if (dist > -0.000001 && pair_list->dist <= (dist - node->rad))
        return;
    else if (node->left == NULL && node->right == NULL) {
        if (dist > pair_list->dist)
            return;
        pair_list->dist = dist;
        pair_list->point = node->pivot;
    
        float f_temp = 0.0;
        datapoint* d_temp = NULL;
        for(int n = 1; n < K && (pair_list + n-1)->dist < (pair_list + n)->dist; n++) {
            f_temp = (pair_list + n)->dist;
            d_temp = (pair_list + n)->point;
            (pair_list + n)->dist = (pair_list + n-1)->dist;
            (pair_list + n)->point = (pair_list + n-1)->point;
            (pair_list + n-1)->dist = f_temp;
            (pair_list + n-1)->point = d_temp;
        }
    } else {
        float leftPivotDist = getDistance(point, node->left->pivot);
        float rightPivotDist = getDistance(point, node->right->pivot);
        float leftBallDist = leftPivotDist - node->left->rad;
        float rightBallDist = rightPivotDist - node->right->rad;
        
        if (leftBallDist < 0 && rightBallDist < 0) {
            if (leftPivotDist < rightPivotDist) {
                k_nearest_neighbor_search(node->left, point, pos);
                k_nearest_neighbor_search(node->right, point, pos);
            } else {
                k_nearest_neighbor_search(node->right, point, pos);
                k_nearest_neighbor_search(node->left, point, pos);
            }
        } else {
            if (leftBallDist < rightBallDist) {
                k_nearest_neighbor_search(node->left, point, pos);
                k_nearest_neighbor_search(node->right, point, pos);
            } else {
                k_nearest_neighbor_search(node->right, point, pos);
                k_nearest_neighbor_search(node->left, point, pos);
            }
        }
    }
}*/

