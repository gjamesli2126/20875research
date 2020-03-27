/* -*- mode: C++ -*- */
/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/

#include "nn.h"

__global__ void init_kernel(void) {
	return;
}


__global__ void search_kernel(struct __GPU_tree d_tree, struct Point *__GPU_point_Point_d, struct __GPU_point *__GPU_point_array_d, struct Point *__GPU_node_point_d, int* index_buffer) {
	
	int pidx, idx;
	//struct Point *target;

//	int sp;
	__shared__ int SP[NUM_OF_WARPS_PER_BLOCK];
	
#define sp SP[WARP_INDEX]

	bool curr, cond, status;
    bool opt1, opt2;
	int critical;
	unsigned int vote_left;
	unsigned int vote_right;
	unsigned int num_left;
	unsigned int num_right;

	int cur_node_index;

	__shared__ int node_stack[NUM_OF_WARPS_PER_BLOCK][64];

	struct __GPU_Node node;
	struct __GPU_Node parent_node; // structs cached into registers

//	__shared__ struct Point target[THREADS_PER_BLOCK]; // point data cached in SMEM
	struct Point target;

//	for (pidx = blockIdx.x * blockDim.x + threadIdx.x; pidx < d_tree.npoints; pidx += blockDim.x * gridDim.x) {
	for (idx = blockIdx.x * blockDim.x + threadIdx.x; idx < d_tree.npoints; idx += blockDim.x * gridDim.x) {
		pidx = index_buffer[idx];
		target = __GPU_point_Point_d[__GPU_point_array_d[pidx].target];
		sp = 0;
		status = 1;
		critical = 63;
		cond = 1;
		node_stack[WARP_INDEX][0] = 0;
		
		while(sp >= 0) {
			
			cur_node_index = node_stack[WARP_INDEX][sp--];

#ifdef TRACK_TRAVERSALS
                target.num_nodes_traversed++;
#endif

			if (status == 0 && critical >= sp) {
				status = 1;
			}
			
			if (status == 1) {
				node = d_tree.nodes[cur_node_index];
				int parent_node_index = node.parent;

				if(parent_node_index != -1) {
					parent_node = d_tree.nodes[parent_node_index];
					float upperDist = 0.0;
					int i;
					struct Point *a = &__GPU_node_point_d[parent_node.point];				
					for(i = 0; i < DIM; i++) {
						float diff = (a->coord[i] - target.coord[i]);
						upperDist += (diff*diff);
					}
					upperDist = sqrt(upperDist);
				
					if(parent_node.right == cur_node_index) {
						cond = upperDist + target.tau >= parent_node.threshold;
						if(!__any(cond)) {
#ifdef TRACK_TRAVERSALS
							target.num_trunc++;
#endif
							continue;
						}
					} else if(parent_node.left == cur_node_index) {
						cond = upperDist - target.tau <= parent_node.threshold;
						if(!__any(cond)) {
#ifdef TRACK_TRAVERSALS
							target.num_trunc++;
#endif
							continue;
						}
					}				
				}

				if (!cond) {
					status = 0;
					critical = sp - 1;
				} else {
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

					opt1 = dist < node.threshold;
					opt2 = dist >= node.threshold;
					vote_left = __ballot(opt1);
					vote_right = __ballot(opt2);
					num_left = __popc(vote_left);
					num_right = __popc(vote_right);
					if(num_left > num_right) {
						if (node.right != -1) {
							node_stack[WARP_INDEX][++sp] = node.right;
						}
						if (node.left != -1) {
							node_stack[WARP_INDEX][++sp] = node.left;
						}
					} else {
						if (node.left != -1) {
							node_stack[WARP_INDEX][++sp] = node.left;
						}
						if (node.right != -1) {
							node_stack[WARP_INDEX][++sp] = node.right;
						}
					}
				}
			}
		}

		__GPU_point_Point_d[__GPU_point_array_d[pidx].target] = target;
	}
#undef sp
}


__global__ void search_pre_kernel(struct __GPU_tree d_tree, struct Point *__GPU_point_Point_d, struct __GPU_point *__GPU_point_array_d, struct Point *__GPU_node_point_d, int* d_matrix, int start, int end, int interval) {
	
	int pidx;
	//struct Point *target;

	int sp;
	int cur_node_index;

	int node_stack[128];

	struct __GPU_Node node;
	struct __GPU_Node parent_node; // structs cached into registers
//	__shared__ struct Point target[THREADS_PER_BLOCK]; // point data cached in SMEM
	struct Point target;

	for (pidx = blockIdx.x * blockDim.x + threadIdx.x + start; pidx < end; pidx += blockDim.x * gridDim.x) {
		
		target = __GPU_point_Point_d[__GPU_point_array_d[pidx].target];
		sp = 0;
		node_stack[0] = 0;
		
		int d_matrix_index = pidx * interval;
/*		for (int i = d_matrix_index; i < d_matrix_index + interval; i ++) {
			d_matrix[i] = -1;
		}*/

		while(sp >= 0) {
			cur_node_index = node_stack[sp--];

			node = d_tree.nodes[cur_node_index];
			if (node.depth == SPLICE_DEPTH)
			{
				d_matrix[d_matrix_index] = node.pre_id - 1;
				d_matrix_index ++;
//				d_matrix[node.pre_id * d_tree.npoints + pidx] = 1;
			}

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

			if (node.depth < SPLICE_DEPTH)
			{
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
		}

//		__GPU_point_Point_d[__GPU_point_array_d[pidx].target] = target[threadIdx.x];
	}
}
