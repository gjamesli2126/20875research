/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __PC_KERNEL_H_
#define __PC_KERNEL_H_

#include <cuda.h>
#include "pc_gpu.h"

__global__ void init_kernel(void);
__global__ void compute_correlation(pc_kernel_params params);

#endif
