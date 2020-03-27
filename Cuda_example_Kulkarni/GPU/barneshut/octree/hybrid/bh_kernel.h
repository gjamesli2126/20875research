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

//#define USE_SMEM
/*
#ifdef USE_SMEM
#define STACK_INIT() sp = 1; stack[sp][WARP_INDEX].items.index = 0; stack[sp][WARP_INDEX].items.dsq = size * size * itolsq
#define STACK_POP() sp -= 1
#define STACK_PUSH() sp += 1
#define STACK_TOP_NODE_INDEX stack[sp][WARP_INDEX].items.index
#define STACK_TOP_DSQ stack[sp][WARP_INDEX].items.dsq
#else
//#define STACK_INIT() sp = 1; stack_node_index[sp] = 0; stack_dsq[sp] = base_dsq
#define STACK_INIT() sp = 1; stack_node_index[sp] = 0; stack_dsq[sp] = size * size * itolsq
#define STACK_POP() sp -= 1
#define STACK_PUSH() sp += 1
#define STACK_TOP_NODE_INDEX stack_node_index[sp]
#define STACK_TOP_DSQ stack_dsq[sp]
#endif

#define CUR_NODE0 cur_node0[WARP_INDEX]
#define CUR_NODE1 cur_node1[WARP_INDEX]
#define CUR_NODE2 cur_node2[WARP_INDEX]
#define CUR_NODE3 cur_node3[WARP_INDEX]
*/
__global__ void init_kernel(void);
__global__ void compute_force_gpu(bh_kernel_params params);
__global__ void compute_force_pre_gpu(bh_kernel_params params, long start, long end);

#endif
