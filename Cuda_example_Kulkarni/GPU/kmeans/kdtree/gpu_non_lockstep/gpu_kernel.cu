/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include "gpu_kernel.h"
#include <float.h>


__global__ void init_kernel(void) {

}

__global__ void nearest_cluster (gpu_tree gpu_tree, DataPoint *points, int npoints, int K) {

	int closest;
	float closestDist;
#ifdef TRACK_TRAVERSALS
	int numNodesTraversed;
#endif

	int i, j, pidx;
	int cur_node_index, prev_node_index, sp;

	int stk[64];
	int stk_top;

	gpu_tree_node_0 cur_node0;
	gpu_tree_node_1 cur_node1;
	gpu_tree_node_2 cur_node2;
	gpu_tree_node_3 cur_node3;

	float dist=0.0;
    int id = 0;
    int axis = 0;
    long base = 0;
    bool* visited;

	for (pidx = blockIdx.x * blockDim.x + threadIdx.x; pidx < npoints; pidx += blockDim.x * gridDim.x) {

		closest = points[pidx].clusterId;
		closestDist = FLT_MAX;
		visited = &gpu_tree.visited[pidx * K];
		base = pidx * K;
		for (i = 0; i < K; i ++) {
			gpu_tree.visited[base + i] = false;
		}
#ifdef TRACK_TRAVERSALS
		numNodesTraversed = points[pidx].numNodesTraversed;
#endif

      	cur_node_index = 0;
      	STACK_INIT ();
      	STACK_TOP_NODE_INDEX = 0;
			
		while(sp >= 1) {
			cur_node_index = STACK_TOP_NODE_INDEX;
			STACK_POP();	

#ifdef TRACK_TRAVERSALS
			numNodesTraversed++;
#endif

			prev_node_index = -1;
			while (cur_node_index != -1) {
				axis = gpu_tree.nodes0[cur_node_index].axis;
				if (axis == DIM || points[pidx].coord[axis] <= gpu_tree.nodes1[cur_node_index].coord[axis]) {
					STACK_PUSH();
					stk[sp] = cur_node_index;
					cur_node_index = gpu_tree.nodes2[cur_node_index].left;
				} else {					
					STACK_PUSH();
					stk[sp] = cur_node_index;
					cur_node_index = gpu_tree.nodes2[cur_node_index].right;
				}
			}
				
			while (sp >= 1 && STACK_TOP_NODE_INDEX != -1) {
				cur_node_index = STACK_TOP_NODE_INDEX;
				STACK_POP();

				int addr = gpu_tree.nodes0[cur_node_index].clusterId;
				if (gpu_tree.visited[base + addr] == false) {
					gpu_tree.visited[base + addr] = true;
					dist = 0;
					for (i = 0; i < DIM; i ++) {
						dist += (points[pidx].coord[i] - gpu_tree.nodes1[cur_node_index].coord[i]) * (points[pidx].coord[i] - gpu_tree.nodes1[cur_node_index].coord[i]);
					}
					dist = sqrt(dist);
					if (dist < closestDist) {
						closestDist = dist;
						closest = gpu_tree.nodes0[cur_node_index].clusterId;
					}

					axis = gpu_tree.nodes0[cur_node_index].axis;
					if (axis == DIM) {
						dist = 0.0f;
					} else {
						dist = points[pidx].coord[axis] - gpu_tree.nodes1[cur_node_index].coord[axis];
						if (dist < 0.0)
							dist = 0.0 - dist;
					}
					
					if (dist < closestDist) {
						if (gpu_tree.nodes2[cur_node_index].left == prev_node_index && gpu_tree.nodes2[cur_node_index].right != -1) {
							STACK_PUSH();
							stk[sp] = gpu_tree.nodes2[cur_node_index].right;
							break;
						} else if (gpu_tree.nodes2[cur_node_index].left != -1) {
							STACK_PUSH();
							stk[sp] = gpu_tree.nodes2[cur_node_index].left;	
							break;
						}
					}
				}
	            prev_node_index = cur_node_index;
	            if (stk[sp] != gpu_tree.nodes0[cur_node_index].parent) {
		            STACK_PUSH();
					stk[sp] = gpu_tree.nodes0[cur_node_index].parent;
				}
			}
		}

		#ifdef TRACK_TRAVERSALS
		d_search_points[pidx].numNodesTraversed = numNodesTraversed;
		#endif
		points[pidx].clusterId = closest;
	}
}
