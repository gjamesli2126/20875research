/* -*- mode: C++ -*- */
/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/

#include "vptree.h"
#include <stdlib.h>
#include <stdio.h>
__global__ void init_kernel(void) {
	return;
}


__global__ void search_kernel(struct __GPU_tree d_tree, struct Point *__GPU_point_Point_d, struct __GPU_point *__GPU_point_array_d, struct Point *__GPU_node_point_d) {
	
	int pidx;
	//struct Point *target;

	int sp;
	int cur_node_index;

	int node_stack[128];

	struct __GPU_Node node;
	struct __GPU_Node parent_node; // structs cached into registers
//	__shared__ struct Point target[THREADS_PER_BLOCK]; // point data cached in SMEM
	struct Point target;

	for (pidx = blockIdx.x * blockDim.x + threadIdx.x; pidx < d_tree.npoints;
       pidx += blockDim.x * gridDim.x) {
		
		target = __GPU_point_Point_d[__GPU_point_array_d[pidx].target];
		sp = 0;
		node_stack[0] = 0;
		
		while(sp >= 0) {
			
			cur_node_index = node_stack[sp--];
//			if(cur_node_index == -1) {
//				continue;
//			}
//            printf("pidx = %d, cur_node_index = %d.\n", pidx, cur_node_index);

			node = d_tree.nodes[cur_node_index];
			
#ifdef TRACK_TRAVERSALS
			target.num_nodes_traversed++;
#endif


			int parent_node_index = node.parent;
//            printf("pidx = %d, parent_node_index = %d.\n", pidx, parent_node_index);

            parent_node = d_tree.nodes[parent_node_index];
			if(parent_node_index != -1) {
				float upperDist = 0.0;
				int i;
				struct Point *a = &__GPU_node_point_d[parent_node.point];				
				for(i = 0; i < DIM; i++) {
					float diff = (a->coord[i] - target.coord[i]);
					upperDist += (diff*diff);
				}
				upperDist = sqrt(upperDist);
				
				if(parent_node.right == cur_node_index) {
					if(upperDist + target.tau < parent_node.threshold) {
#ifdef TRACK_TRAVERSALS
						target.num_trunc++;
#endif
						continue;
					}
				} else if(parent_node.left == cur_node_index) {
					if(upperDist - target.tau > parent_node.threshold) {
#ifdef TRACK_TRAVERSALS
						target.num_trunc++;
#endif
						continue;
					}
				}				
			}

/*			if(pidx == 0) {
				printf("zero's visiting node %d \n", cur_node_index);
			}
*/
			float dist = 0.0;
			int i;
			struct Point *a = &__GPU_node_point_d[node.point];				
			for(i = 0; i < DIM; i++) {
				float diff = (a->coord[i] - target.coord[i]);
				dist += diff * diff;
			}
			dist = sqrt(dist);

			if(dist < target.tau) {
				target.closest_label = __GPU_node_point_d[node.point].label;
				target.tau = dist;
			}

			int left = node.left; // cache to registers (CSE)
			int right = node.right;
			if(left == -1 && right == -1) {
#ifdef TRACK_TRAVERSALS
				target.num_trunc++;
#endif
				continue;
			}

			if(dist < node.threshold) {
				if (right != -1) {
                    node_stack[++sp] = right;
                }
                if (left != -1) {
				    node_stack[++sp] = left;
                }
			} else {
                if (left != -1) {
				    node_stack[++sp] = left;
                }
                if (right != -1) {
				    node_stack[++sp] = right;
                }
			}
		}

		__GPU_point_Point_d[__GPU_point_array_d[pidx].target] = target;
	}
}
