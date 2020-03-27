/* -*- mode: c -*- */
/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/

#include <float.h>
#include "knn.h"
#include "knn_gpu.h"
#include <stdio.h>

__global__ void init_kernel(void) {

}

__global__ void
nearest_neighbor_search (gpu_tree gpu_tree, int nsearchpoints, node *d_search_points, float *d_nearest_distance,
						int *d_nearest_point_index, int K, int* index_buffer
#ifdef TRACK_TRAVERSALS
						 , int *d_nodes_accessed
#endif
){

	float search_points[DIM];
	float nearest_distance[8];
	int nearest_point_index[8]; 
	float stk_axis_dist[64];

	int pidx;
	int i, j;

	gpu_tree_node_0 cur_node0;
	gpu_tree_node_1 cur_node1;
	gpu_tree_node_2 cur_node2;

//	__shared__ gpu_tree_node_0 cur_node0[NUM_WARPS_PER_BLOCK];
//	__shared__ gpu_tree_node_1 cur_node1[NUM_WARPS_PER_BLOCK];
//	__shared__ gpu_tree_node_2 cur_node2[NUM_WARPS_PER_BLOCK];
	__shared__ int stk_node[NUM_OF_WARPS_PER_BLOCK][64];

	bool cond, status;
    bool opt1, opt2;
	int critical;
	unsigned int vote_left;
	unsigned int vote_right;
	unsigned int num_left;
	unsigned int num_right;

	int cur_node_index;
	__shared__ unsigned int SP[NUM_OF_WARPS_PER_BLOCK];

	int current_split;
	float axis_dist;
	float dist;
	float t;

	float tmpdist;
	int tmpidx;
	int n;

	#ifdef TRACK_TRAVERSALS
	int nodes_accessed;
	#endif


#include "knn_kernel_macros.inc"
	
//  for (pidx = blockIdx.x * blockDim.x + threadIdx.x; pidx < nsearchpoints; pidx += blockDim.x * gridDim.x)
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < nsearchpoints; idx += blockDim.x * gridDim.x)
	{
		pidx = index_buffer[idx];
		for (j = 0; j < DIM; j++) {
				search_points[j] = d_search_points[pidx].point[j];

		}
		
		for(i = 0; i < K; i++) {
			nearest_point_index[i] = -1;
			nearest_distance[i] = FLT_MAX;
		}
		 
		#ifdef TRACK_TRAVERSALS
		nodes_accessed = 0;
		#endif

    	// run this for some number of iterations until done...
    	STACK_INIT();
		status = 1;
		critical = 63;
		cond = 1;
		STACK_NODE=0;
		STACK_AXIS_DIST = FLT_MIN;
		
		while(sp >= 1) {
			// get top of stack
			cur_node_index = STACK_NODE;
			axis_dist = STACK_AXIS_DIST;

#ifdef TRACK_TRAVERSALS
            nodes_accessed++;
#endif

			if (status == 0 && critical >= sp) {
				status = 1;
			}

			STACK_POP();

			if (status) {
				cond = (axis_dist <= nearest_distance[0]);
//			} else {
//				cond = 0;
//			}

			if (!__any(cond)) {
				continue;
			}

//			if (status) {
				if (!cond) {
					status = 0;
					critical = sp;
				} else {
					cur_node0 = gpu_tree.nodes0[cur_node_index];
					current_split = AXIS;
					cur_node1 = gpu_tree.nodes1[cur_node_index];
					cur_node2 = gpu_tree.nodes2[cur_node_index];

					dist = 0.0;
					for (i = 0; i < DIM; i++) {
						t = (POINT[i] - search_points[i]);
						dist +=  t*t;
					}

					// update closest point:
					if(dist < nearest_distance[0]) {
						nearest_distance[0] = dist;
						nearest_point_index[0] = POINT_INDEX;

						// push the value back to maintain sorted order
						for(n = 0; n < K-1 && nearest_distance[n] < nearest_distance[n+1]; n++) {
							tmpdist = nearest_distance[n];
							tmpidx = nearest_point_index[n];
							nearest_distance[n] = nearest_distance[n+1];
							nearest_point_index[n] = nearest_point_index[n+1];
							nearest_distance[n+1] = tmpdist;
							nearest_point_index[n+1] = tmpidx;
						}
					}

					opt1 = search_points[current_split] <= POINT_SPLIT(current_split);
					opt2 = search_points[current_split] > POINT_SPLIT(current_split);
					vote_left = __ballot(opt1);
					vote_right = __ballot(opt2);
					num_left = __popc(vote_left);
					num_right = __popc(vote_right);

					axis_dist =	(search_points[current_split] - POINT_SPLIT (current_split));
					if ((num_left > num_right) && LEFT != NULL_NODE) {
						if(RIGHT != NULL_NODE) {
							STACK_PUSH();
							STACK_NODE = RIGHT;
							if(opt1)
								STACK_AXIS_DIST = axis_dist * axis_dist;
							else
								STACK_AXIS_DIST = FLT_MIN;

						}

						STACK_PUSH();
						STACK_NODE = LEFT;
						STACK_AXIS_DIST = FLT_MIN;

					} else if (RIGHT != NULL_NODE) {
						if(LEFT != NULL_NODE) {
							STACK_PUSH();
							STACK_NODE = LEFT;
							if(opt2)
								STACK_AXIS_DIST = axis_dist * axis_dist;
							else
								STACK_AXIS_DIST = FLT_MIN;
						}

						STACK_PUSH();
						STACK_NODE = RIGHT;
						STACK_AXIS_DIST = FLT_MIN;
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
		d_nodes_accessed[pidx] = nodes_accessed;
		#endif
    }
}

__device__ float distance (float *a, float *b)
{
  int i;
  float d = 0;
  // returns distance squared
#pragma unroll
  for (i = 0; i < DIM; i++)
    {
      d += distance_axis (a, b, i);
    }

  return d;
}



