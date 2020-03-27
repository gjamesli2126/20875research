/* -*- mode: c -*- */
/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/

#include "pc_gpu.h"
#include "pc_kernel.h"
#include "pc_kernel_mem.h"

__global__ void init_kernel(void) {
	
}

__global__ void compute_correlation(pc_kernel_params params) {
	
	int sp;
	float rad;
	int pidx;
	int cur_node_index;

	float p_coord[DIM];
	gpu_node0 cur_node0;
	gpu_node1 cur_node1;
	gpu_node2 cur_node2;
  int stack[128];

	int i, j;
	int can_correlate_result;
	float dist, sum, boxsum, boxdist, center;
	int p_corr;

	int id_test = 0;
	int id_complete = 0;

	#ifdef TRACK_TRAVERSALS
	int p_nodes_accessed;
	#endif

	rad = params.rad;
	for(i = blockIdx.x*blockDim.x + threadIdx.x; i < params.npoints; i+= gridDim.x*blockDim.x) {
		
		pidx = params.points[i];
		p_corr = 0; // params.tree.nodes[pidx].corr;
		
		#ifdef TRACK_TRAVERSALS
		p_nodes_accessed = 0;
		#endif

		for(j = 0; j < DIM; j++) {
			p_coord[j] = params.tree.nodes0[pidx].coord_max[j];
		}

		STACK_INIT();

		while(sp >= 1) {

			cur_node_index = STACK_TOP_NODE_INDEX;
			CUR_NODE0 = params.tree.nodes0[cur_node_index];
			
			#ifdef TRACK_TRAVERSALS
			p_nodes_accessed++;
			#endif

			STACK_POP();

//			if (i == 0) {
//				printf("Test: id = %d, index = %d\n", id_test++, cur_node_index);
//			}
			id_test ++;
			
			// inline call: can_correlate(...)
			sum = 0.0;
			boxsum = 0.0;
			for(j = 0; j < DIM; j++) {
				center = (CUR_NODE0.coord_max[j] + CUR_NODE0.min[j]) / 2;
				boxdist = (CUR_NODE0.coord_max[j] - CUR_NODE0.min[j]) / 2;
				dist = p_coord[j] - center;
				sum += dist * dist;
				boxsum += boxdist * boxdist;
			}

			if(sqrt(sum) - sqrt(boxsum) < rad) { 
				can_correlate_result = 1;
			} else {
				can_correlate_result = 0;
			}

			if(can_correlate_result) {


//				if (i == 0) {
//					printf("	Complete: id = %d, index = %d\n", id_complete++, cur_node_index);
//				}
				id_complete++;

				CUR_NODE1 = params.tree.nodes1[cur_node_index];
				if(CUR_NODE1.splitType == SPLIT_LEAF) {
					// inline call: in_radii(...)
						dist = 0.0;          
						for(j = 0; j < DIM; j++) {
							dist += (p_coord[j] - CUR_NODE0.coord_max[j]) * (p_coord[j] - CUR_NODE0.coord_max[j]);							
						}
						
						dist = sqrt(dist);
						if(dist < rad) {
							p_corr++; // = (100 * block_node_coord[0][WARP_INDEX][k]);
						}

				} else {
					CUR_NODE2 = params.tree.nodes2[cur_node_index];
					// push children
					if(CUR_NODE2.right != -1) {
						STACK_PUSH();
						STACK_TOP_NODE_INDEX = CUR_NODE2.right;
					} 
				
					if(CUR_NODE2.left != -1) {
						STACK_PUSH();
						STACK_TOP_NODE_INDEX = CUR_NODE2.left;
					}

				}
			} else {
				continue;
			} 
		}
		
		params.tree.nodes3[pidx].corr = p_corr;
		#ifdef TRACK_TRAVERSALS
		params.tree.nodes0[pidx].nodes_accessed = p_nodes_accessed;
		#endif
//		printf("point id = %d, max[0] = %f, test = %d, complete = %d\n", params.tree.nodes1[pidx].id, params.tree.nodes0[pidx].coord_max[0], id_test, id_complete);
	}
}
