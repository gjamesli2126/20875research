/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifdef USE_SMEM
#define STACK_INIT() sp = 1; stack[sp][WARP_INDEX].items.index = 0; stack[sp][WARP_INDEX].items.dsq = size * size * itolsq
#define STACK_POP() sp -= 1
#define STACK_PUSH() sp += 1
#define STACK_TOP_NODE_INDEX stack[sp][WARP_INDEX].items.index
#define STACK_TOP_DSQ stack[sp][WARP_INDEX].items.dsq
#else
#define STACK_INIT() sp = 1; stack[WARP_INDEX][sp].items.index = 0; stack[WARP_INDEX][sp].items.dsq = size * size * itolsq
#define STACK_POP() sp -= 1
#define STACK_PUSH() sp += 1
#define STACK_TOP_NODE_INDEX stack[WARP_INDEX][sp].items.index
#define STACK_TOP_DSQ stack[WARP_INDEX][sp].items.dsq
#endif

#define CUR_NODE0 cur_node0
#define CUR_NODE1 cur_node1
#define CUR_NODE2 cur_node2
#define CUR_NODE3 cur_node3
