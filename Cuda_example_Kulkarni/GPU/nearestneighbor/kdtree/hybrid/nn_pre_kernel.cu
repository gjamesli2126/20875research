/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include "nn_pre_kernel.h"

/*#define STACK_INIT()	sp = 1;
#define STACK_PUSH(node) sp = sp + 1; stk[threadIdx.x][sp] = node
#define STACK_POP() sp = sp - 1;
#define STACK_NODE stk[threadIdx.x][sp]*/

#define STACK_INIT()																										\
	sp = 0;																																\
	stk = &d_tree.stk[d_tree.depth*blockIdx.x*blockDim.x + threadIdx.x];  \
	stk_top = 0;

#define STACK_PUSH(node) sp = sp + 1; *stk = stk_top; stk_top = node; stk += blockDim.x;

#define STACK_POP() sp = sp - 1; stk -= blockDim.x; if(sp >= 0) { stk_top = *stk; }

__global__ void pre_nearest_neighbor_search (kernel_params params, int *d_matrix, int start, int end, int interval)
{
 	float search_points_coord[DIM];
	int closest;
	float closestDist;

	#ifdef TRACK_TRAVERSALS
	int numNodesTraversed;
	#endif

	int i, j, pidx;

	gpu_tree d_tree = params.d_tree;
    gpu_point *d_training_points = params.d_training_points;
    int n_training_points = params.n_training_points;
    gpu_point *d_search_points = params.d_search_points;
    int n_search_points = params.n_search_points;
    int *d_array_points = params.d_array_points;
	int seg_index = d_tree.max_nnodes / 2; 
//    int n_array_points = params.n_array_points;
	
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

//#include "nn_kernel_macros.inc"

	for (pidx = blockIdx.x * blockDim.x + threadIdx.x + start; pidx < end; pidx += blockDim.x * gridDim.x)
    {
		for(j = 0; j < DIM; j++) {
			search_points_coord[j] = d_search_points[pidx].coord[j];
		}	

		closest = d_search_points[pidx].closest;
		closestDist = d_search_points[pidx].closest_dist;
		#ifdef TRACK_TRAVERSALS
		numNodesTraversed = d_search_points[pidx].num_nodes_traversed;
		#endif

		cur_node_index = 0;

		STACK_INIT();
		
        stk_top = params.n_root_index;
		int d_matrix_index = pidx * interval;
			
		while(sp >= 0) {
			cur_node_index = stk_top;
			if(d_tree.nodes0[cur_node_index].items.depth == SPLICE_DEPTH)
			{
				d_matrix[d_matrix_index] = d_tree.nodes0[cur_node_index].items.pre_id - 1;
				d_matrix_index ++;
			}
			
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
				float max = d_tree.nodes1[cur_node_index].items[i].max;
				float min = d_tree.nodes1[cur_node_index].items[i].min;
				center = (max + min) / 2;
				boxdist = (max - min) / 2;
				dist = search_points_coord[i] - center;
				sum += dist * dist;
				boxsum += boxdist * boxdist;
			}

			if(sqrt(sum) - sqrt(boxsum) < sqrt(closestDist)) {
				cur_node0 = d_tree.nodes0[cur_node_index];
				if(cur_node0.items.axis == DIM) {
					cur_node3 = d_tree.nodes3[cur_node_index];
					for(i = 0; i < MAX_POINTS_IN_CELL; i++) {
						if(cur_node3.items.points[i] >= 0) {
							// update closest...
							float dist = 0.0;
							float t;

							for(j = 0; j < DIM; j++) {
								t = (d_training_points[cur_node3.items.points[i]].coord[j] - search_points_coord[j]);
								dist += t*t;
							}

							if(dist <= closestDist) {
								closest = cur_node3.items.points[i];
								closestDist = dist;
							}
						}
					}	
				} else {
					cur_node2 = d_tree.nodes2[cur_node_index];
					if (d_tree.nodes0[cur_node_index].items.depth < SPLICE_DEPTH)
					{
						if (search_points_coord[cur_node0.items.axis] < cur_node0.items.splitval) {
							
							if(RIGHT != NULL_NODE) { STACK_PUSH(RIGHT); }
							if(LEFT != NULL_NODE) { STACK_PUSH(LEFT); }
						} else {
							if(LEFT != NULL_NODE) { STACK_PUSH(LEFT); }
							if(RIGHT != NULL_NODE) { STACK_PUSH(RIGHT); }
						}
					}
				}				
			}
		}

//		d_search_points[pidx].closest = closest;
//		d_search_points[pidx].closestDist = closestDist;
//		#ifdef TRACK_TRAVERSALS
//		d_search_points[pidx].numNodesTraversed = numNodesTraversed;
//		#endif
	}

}


