/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __BH_KERNEL_H
#define __BH_KERNEL_H

#include "bh.h"
#include "bh_gpu.h"
#include "bh_block.h"
#include "bh_kernel_params.h"

// warp index within block
#define WARP_INDEX (threadIdx.x >> 5)

// warp index across all blocks
#define GLOBAL_WARP_INDEX (WARP_INDEX + (blockIdx.x*NWARPS_PER_BLOCK))

// index within the warp
#define THREAD_INDEX_IN_WARP (threadIdx.x - (THREADS_PER_WARP * WARP_INDEX))

// true if thread is the first in the warp
#define IS_FIRST_THREAD_IN_WARP (threadIdx.x == (WARP_INDEX * THREADS_PER_WARP))

#ifndef USE_LOCAL_STACK
#define STACK_INIT() sp = 1; \
	stack_node_index = &params.root.stk_nodes[params.root.depth*8*blockIdx.x*blockDim.x + threadIdx.x]; \
	stack_dsq = &params.root.stk_dsq[params.root.depth*8*blockIdx.x*blockDim.x + threadIdx.x]; \
	*stack_node_index = 0; *stack_dsq = size * size * itolsq

#define STACK_POP() sp -= 1; stack_node_index -= blockDim.x; stack_dsq -= blockDim.x
#define STACK_PUSH() sp += 1; stack_node_index += blockDim.x; stack_dsq += blockDim.x
#define STACK_TOP_NODE_INDEX (*stack_node_index)
#define STACK_TOP_DSQ (*stack_dsq)
#else
#define STACK_INIT() sp = 1; stack_node_index[sp] = 0; stack_dsq[sp] = size * size * itolsq
#define STACK_POP() sp -= 1
#define STACK_PUSH() sp += 1
#define STACK_TOP_NODE_INDEX stack_node_index[sp]
#define STACK_TOP_DSQ stack_dsq[sp]
#endif

#define CUR_NODE0 cur_node0
#define CUR_NODE1 cur_node1
#define CUR_NODE2 cur_node2
#define CUR_NODE3 cur_node3

__global__ void init_kernel(void);
__global__ void compute_force_gpu(bh_kernel_params params);

#endif
