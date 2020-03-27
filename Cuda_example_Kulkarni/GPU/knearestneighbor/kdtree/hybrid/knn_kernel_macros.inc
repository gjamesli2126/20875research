/* -*- mode: c -*- */
/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/

#define STACK_AXIS_DIST stk_axis_dist[sp]
#define STACK_NODE stk_node[WARP_INDEX][sp]

#define AXIS cur_node0.items.axis
#define POINT cur_node2.point
#define POINT_INDEX cur_node0.items.point_index
#define POINT_SPLIT(s) cur_node2.point[s]
//cur_node2.point[s]
#define LEFT cur_node1.left
#define RIGHT cur_node1.right

#define sp SP[WARP_INDEX]

#define STACK_INIT() sp = 1;

//stk_node = &gpu_tree.stk_node[gpu_tree.depth*2*blockIdx.x*blockDim.x + threadIdx.x]; \
//	stk_axis_dist = &gpu_tree.stk_axis_dist[gpu_tree.depth*2*blockIdx.x*blockDim.x + threadIdx.x]; \
// stk_node_top = 0;																										\
//	stk_axis_dist_top = FLT_MIN;

#define STACK_PUSH() sp = sp + 1;												
#define STACK_POP() sp = sp - 1; 

