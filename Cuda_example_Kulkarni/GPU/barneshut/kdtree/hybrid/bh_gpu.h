/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __BH_GPU_H
#define __BH_GPU_H

#include <cuda.h>
#include "bh.h"

#ifndef NUM_OF_WARPS_PER_BLOCK
#define NUM_OF_WARPS_PER_BLOCK 			6
#endif 

#ifndef NUM_OF_BLOCKS
#define NUM_OF_BLOCKS 					(1024)
#endif 

#define NUM_OF_THREADS_PER_WARP 		32
#define NUM_OF_THREADS_PER_BLOCK 		NUM_OF_WARPS_PER_BLOCK * NUM_OF_THREADS_PER_WARP

#define BLOCK_SIZE (1)

typedef struct _bh_kernel_stacks {
	int * block_stack;
	// Barnes hut is nice in that we actually do not need any stack
	// space for local variables.

} bh_kernel_stacks;

#endif
