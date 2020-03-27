/* -*- mode: c -*- */
/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/

#include <float.h>
#include "knn.h"
#include "knn_gpu.h"
#include <stdio.h>

extern __device__ float distance (float *a, float *b);

__global__ void
nearest_neighbor_pre_search (gpu_tree gpu_tree, int nsearchpoints, node *d_search_points, float *d_nearest_distance,
						int *d_nearest_point_index, int K, int* d_matrix, int start, int end, int interval
#ifdef TRACK_TRAVERSALS
						 , int *d_nodes_accessed
#endif
)
{
	extern __shared__ int smem[];
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

	int current_split;
	float axis_dist;
	float dist;
	float t;

  // Get the position of the 1st item
	int *stk_node;
	int stk_node_top;
	float *stk_axis_dist;
	float stk_axis_dist_top;

	float tmpdist;
	int tmpidx;
	int n;

	#ifdef TRACK_TRAVERSALS
	int nodes_accessed;
	#endif

#include "knn_pre_kernel_macros.inc"
	
	for (pidx = blockIdx.x * blockDim.x + threadIdx.x + start; pidx < end; pidx += blockDim.x * gridDim.x)
    {
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
		int d_matrix_index = pidx * interval;
		while(sp >= 0) {
			// get top of stack
			cur_node_index = STACK_NODE;
			axis_dist = STACK_AXIS_DIST;

			if (gpu_tree.nodes0[cur_node_index].items.depth == SPLICE_DEPTH)
			{
				d_matrix[d_matrix_index] = gpu_tree.nodes0[cur_node_index].items.pre_id - 1;
				d_matrix_index ++;
			}

			if(axis_dist > nearest_distance[0]) {
				STACK_POP();
				continue;
			}
				
			#ifdef TRACK_TRAVERSALS
			nodes_accessed++;
			#endif

			cur_node0 = gpu_tree.nodes0[cur_node_index];
			current_split = AXIS;
										
			// Swap it if our point is closer
			//dist = distance (POINT, &search_points[threadIdx.x * DIM]);
			dist = 0.0;
			for (i = 0; i < DIM; i++) {
				t = (gpu_tree.nodes2[cur_node_index].point[i] - search_points[i]);
				dist +=  t*t;
			}

			// update closest point:
			if(dist < nearest_distance[0]) {
				nearest_distance[0] = dist;
				nearest_point_index[0] = POINT_INDEX;
							
				// push the value back to maintain sorted order
				for(n = 0; n < K-1 && nearest_distance[n] < nearest_distance[n+1]; n++) {
					tmpdist = nearest_distance[n+1];
					tmpidx = nearest_point_index[n+1];
					nearest_distance[n+1] = nearest_distance[n];
					nearest_point_index[n+1] = nearest_point_index[n];
					nearest_distance[n] = tmpdist;
					nearest_point_index[n] = tmpidx;
				}
			}							

			cur_node1 = gpu_tree.nodes1[cur_node_index];
			if (LEFT != NULL_NODE && search_points[current_split] <= POINT_SPLIT (current_split)) {
				axis_dist =	(search_points[current_split] - POINT_SPLIT (current_split));
				if(RIGHT != NULL_NODE && cur_node0.items.depth < SPLICE_DEPTH) {
					STACK_NODE = RIGHT;
					STACK_AXIS_DIST = axis_dist * axis_dist;						
					STACK_PUSH();
				}
					
				STACK_NODE = LEFT;
				STACK_AXIS_DIST = FLT_MIN;					
			} else if (RIGHT != NULL_NODE) {
				axis_dist =	(search_points[current_split] - POINT_SPLIT (current_split));
					
				if(LEFT != NULL_NODE && cur_node0.items.depth < SPLICE_DEPTH) {	
					STACK_NODE = LEFT;
					STACK_AXIS_DIST = axis_dist * axis_dist;
					STACK_PUSH();
				}

				STACK_NODE = RIGHT;
				STACK_AXIS_DIST = FLT_MIN;
			} else {
				STACK_POP();
			}
		}
			
		// Save to global memory
/*		for(i = 0; i < K; i++) {
			d_nearest_point_index[K*pidx+i] = nearest_point_index[i*NUM_THREADS_PER_BLOCK+threadIdx.x];
			d_nearest_distance[K*pidx+i] = nearest_distance[i*NUM_THREADS_PER_BLOCK+threadIdx.x];
		}
		#ifdef TRACK_TRAVERSALS
		d_nodes_accessed[pidx] = nodes_accessed;
		#endif*/
    }
}



