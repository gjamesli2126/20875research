/* -*- mode: c -*- */
/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/

#include <float.h>
#include "nn.h"
#include "nn_gpu.h"
#include "gpu_tree.h"

__global__ void init_kernel(void) {

}

__global__ void nearest_neighbor_search (gpu_tree gpu_tree, gpu_point *d_training_points, int n_training_points,
																				 gpu_point *d_search_points, int n_search_points) 
{

 	__shared__ float search_points_coord[NUM_THREADS_PER_BLOCK*DIM];
	int closest;
	float closestDist;

	#ifdef TRACK_TRAVERSALS
	int numNodesTraversed;
	#endif

	int i, j, pidx;
	
	int cur_node_index, prev_node_index, sp;

	int *stk;
	int stk_top;

	gpu_tree_node_0 cur_node0;
  //gpu_tree_node_1 cur_node1;
	gpu_tree_node_2 cur_node2;
	gpu_tree_node_3 cur_node3;

	float dist=0.0;
	float boxdist=0.0;
	float sum=0.0;
	float boxsum=0.0;
	float center=0.0;
    int id = 0;

#include "nn_kernel_macros.inc"

	for (pidx = blockIdx.x * blockDim.x + threadIdx.x; pidx < n_search_points;
       pidx += blockDim.x * gridDim.x)
    {
			for(j = 0; j < DIM; j++) {
				search_points_coord[j*NUM_THREADS_PER_BLOCK + threadIdx.x] = d_search_points[pidx].coord[j];
			}	

			closest = d_search_points[pidx].closest;
			closestDist = d_search_points[pidx].closestDist;
			#ifdef TRACK_TRAVERSALS
			numNodesTraversed = d_search_points[pidx].numNodesTraversed;
			#endif

      cur_node_index = 0;

      STACK_INIT ();
			
			while(sp >= 0) {
				cur_node_index = stk_top;
				
				STACK_POP();
				#ifdef TRACK_TRAVERSALS
				numNodesTraversed++;
				#endif

				//cur_node1 = gpu_tree.nodes1[cur_node_index];				
				
				// inlined function can_correlate
				dist=0.0;
				boxdist=0.0;
				sum=0.0;
				boxsum=0.0;
				center=0.0;

				for(i = 0; i < DIM; i++) {
					float max = gpu_tree.nodes1[cur_node_index].items.max[i];
					float min = gpu_tree.nodes1[cur_node_index].items.min[i];
					center = (max + min) / 2;
					boxdist = (max - min) / 2;
					dist = search_points_coord[i*NUM_THREADS_PER_BLOCK + threadIdx.x] - center;
					sum += dist * dist;
					boxsum += boxdist * boxdist;
				}

				if(sqrt(sum) - sqrt(boxsum) < sqrt(closestDist)) {
					cur_node0 = gpu_tree.nodes0[cur_node_index];
					if(cur_node0.items.axis == DIM) {
						cur_node3 = gpu_tree.nodes3[cur_node_index];
						for(i = 0; i < MAX_POINTS_IN_CELL; i++) {
							if(cur_node3.points[i] >= 0) {
								// update closest...
								float dist = 0.0;
								float t;

								for(j = 0; j < DIM; j++) {
									t = (d_training_points[cur_node3.points[i]].coord[j] - search_points_coord[j*NUM_THREADS_PER_BLOCK + threadIdx.x]);
									dist += t*t;
								}

								if(dist <= closestDist) {
									closest = cur_node3.points[i];
									closestDist = dist;
								}
							}
						}	
					} else {
						cur_node2 = gpu_tree.nodes2[cur_node_index];
						if (search_points_coord[cur_node0.items.axis*NUM_THREADS_PER_BLOCK + threadIdx.x] < cur_node0.items.splitval) {
							
							if(RIGHT != NULL_NODE) { STACK_PUSH(RIGHT); }
							if(LEFT != NULL_NODE) { STACK_PUSH(LEFT); }
						} else {
							if(LEFT != NULL_NODE) { STACK_PUSH(LEFT); }
							if(RIGHT != NULL_NODE) { STACK_PUSH(RIGHT); }
						}
					}				
				}
			}

			d_search_points[pidx].closest = closest;
			d_search_points[pidx].closestDist = closestDist;
			#ifdef TRACK_TRAVERSALS
			d_search_points[pidx].numNodesTraversed = numNodesTraversed;
			#endif

		}
}
 
