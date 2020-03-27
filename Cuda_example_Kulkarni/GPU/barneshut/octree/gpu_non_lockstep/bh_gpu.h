/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __BH_GPU_H
#define __BH_GPU_H

#include <cuda.h>
#include "bh.h"

#define NWARPS 6
#define NUM_THREAD_BLOCKS (1024)
#define THREADS_PER_WARP 32
#define THREADS_PER_BLOCK NWARPS*THREADS_PER_WARP
#define NWARPS_PER_BLOCK (THREADS_PER_BLOCK / THREADS_PER_WARP)

#define BLOCK_SIZE (1)

typedef struct _bh_kernel_stacks {
	int * block_stack;
	// Barnes hut is nice in that we actually do not need any stack
	// space for local variables.

} bh_kernel_stacks;

#endif
