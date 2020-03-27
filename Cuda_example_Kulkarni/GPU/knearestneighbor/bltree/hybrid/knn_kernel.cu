/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
//
//  BBT_kenrel.cpp
//  BallTree-kNN

#include "knn_kernel.h"

#define WARP_INDEX (threadIdx.x >> 5)
#define STACK_NODE stk_node[WARP_INDEX][sp]
//#define POINT cur_node2.point
#define POINT_INDEX cur_node1[WARP_INDEX].idx
//cur_node2.point[s]
#define LEFT cur_node1[WARP_INDEX].left
#define RIGHT cur_node1[WARP_INDEX].right

#define sp SP[WARP_INDEX]

#define STACK_INIT() \
	sp = 1;	

#define STACK_PUSH() sp = sp + 1; 

#define STACK_POP() sp = sp - 1; 

__global__ void init_kernel(void) {

}

__global__ void k_nearest_neighbor_search (gpu_tree gpu_tree, int nsearchpoints, datapoint *d_search_points,
											float *d_nearest_distance, int *d_nearest_point_index, int K, 
                                            int* index_buffer) {
	float search_points[DIM];
	float nearest_distance[8];
	int nearest_point_index[8];

	int pidx;
	int i, j;

	int cur_node_index;
//	int sp;
	
#ifdef TRACK_TRAVERSALS
	int numNodesTraversed;
#endif

	// Get the position of the 1st item
	__shared__ gpu_tree_node_0 cur_node0[NUM_OF_WARPS_PER_BLOCK];
	__shared__ gpu_tree_node_1 cur_node1[NUM_OF_WARPS_PER_BLOCK];
	__shared__ int stk_node[NUM_OF_WARPS_PER_BLOCK][64];

	bool curr, cond, status;
//    bool opt1, opt2;
	unsigned int critical;
	__shared__ unsigned int vote_left;
	__shared__ unsigned int vote_right;
//	__shared__ unsigned int num_left;
//	__shared__ unsigned int num_right;
	__shared__ unsigned int SP[NUM_OF_WARPS_PER_BLOCK];

	float tmpdist;
	int tmpidx;
	int n;
	float t;
	float dist = 0.0;
	float leftPivotDist = 0.0;
	float rightPivotDist = 0.0;

	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < nsearchpoints; idx += blockDim.x * gridDim.x) {
		pidx = index_buffer[idx];
        for (j = 0; j < DIM; j++)
			search_points[j] = d_search_points[pidx].coord[j];

		for(i = 0; i < K; i++) {
			nearest_point_index[i] = -1;
			nearest_distance[i] = FLT_MAX;
		}
#ifdef TRACK_TRAVERSALS
		numNodesTraversed = 0; //d_search_points[pidx].numNodesTraversed;
#endif

		// run this for some number of iterations until done...
		STACK_INIT();
		STACK_NODE = 0;
		status = 1;
		critical = 63;
		cond = 1;
		while(sp >= 1) {
			// get top of stack
			cur_node_index = STACK_NODE;

#ifdef TRACK_TRAVERSALS
            numNodesTraversed++;
#endif

			if (critical >= sp)
				status = 1;
			
			if (status) {			
				dist = 0.0;
				for (i = 0; i < DIM; i++) {
					t = (gpu_tree.nodes0[cur_node_index].coord[i] - search_points[i]);
					dist +=  t*t;
				}
				dist = sqrt(dist);
				cond =  dist < -0.000001 || nearest_distance[0] > (dist - gpu_tree.nodes0[cur_node_index].rad);
//			}
			STACK_POP();
			if (!__any(cond)) {
				continue;
			}
			
//			if (status) {
				if (!cond) {
					status = 0;
					critical = sp;
				} else {
					cur_node1[WARP_INDEX] = gpu_tree.nodes1[cur_node_index];

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
						}
						for (i = 0; i < DIM; i++) {
							t = (gpu_tree.nodes0[RIGHT].coord[i] - search_points[i]);
							rightPivotDist +=  t*t;
						}

						vote_left = __ballot(leftPivotDist < rightPivotDist);
						vote_right = __ballot(leftPivotDist >= rightPivotDist);
						if (__popc(vote_left) > __popc(vote_right)) {
							STACK_PUSH();
							STACK_NODE = RIGHT;
							STACK_PUSH();
							STACK_NODE = LEFT;
						} else {
							STACK_PUSH();
							STACK_NODE = LEFT;
							STACK_PUSH();
							STACK_NODE = RIGHT;
						}
					}
				}
			}
		}

		// Save to global memory
		for(i = 0; i < K; i++) {
			d_nearest_point_index[K*pidx+i] = nearest_point_index[i];
			d_nearest_distance[K*pidx+i] = nearest_distance[i];
		}
#ifdef TRACK_TRAVERSALS
		d_search_points[pidx].numNodesTraversed = numNodesTraversed;
#endif
	}
}


