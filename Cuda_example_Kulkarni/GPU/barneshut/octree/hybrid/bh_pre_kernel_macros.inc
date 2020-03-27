/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifdef STACK_INIT()
#undef STACK_INIT()
#undef STACK_POP()
#undef STACK_PUSH()
#undef STACK_TOP_NODE_INDEX
#undef STACK_TOP_DSQ
#endif

#ifdef sp
#undef sp
#endif

#define STACK_INIT() sp = 1; stack_node_index[sp] = 0; stack_dsq[sp] = size * size * itolsq
#define STACK_POP() sp -= 1
#define STACK_PUSH() sp += 1
#define STACK_TOP_NODE_INDEX stack_node_index[sp]
#define STACK_TOP_DSQ stack_dsq[sp]

#define CUR_NODE0 cur_node0
#define CUR_NODE1 cur_node1
#define CUR_NODE2 cur_node2
#define CUR_NODE3 cur_node3
