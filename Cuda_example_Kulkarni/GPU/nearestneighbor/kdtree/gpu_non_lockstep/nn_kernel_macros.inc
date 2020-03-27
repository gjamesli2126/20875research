/* -*- mode: c -*- */
/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/

#define LEFT cur_node2.items.left
#define RIGHT cur_node2.items.right

#define STACK_INIT()																										\
	sp = 0;																																\
	stk = &gpu_tree.stk[gpu_tree.depth*blockIdx.x*blockDim.x + threadIdx.x];  \
	stk_top = 0;

#define STACK_PUSH(node) sp = sp + 1; *stk = stk_top; stk_top = node; stk += blockDim.x;

#define STACK_POP() sp = sp - 1; stk -= blockDim.x; if(sp >= 0) { stk_top = *stk; }
