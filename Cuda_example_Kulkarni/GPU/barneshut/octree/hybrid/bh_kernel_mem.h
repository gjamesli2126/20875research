#ifndef __BH_KERNEL_MEM_H
#define __BH_KERNEL_MEM_H

#include <cuda.h>

#include "bh.h"
#include "bh_gpu.h"
#include "bh_gpu_tree.h"
#include "bh_kernel_params.h"

void allocate_gpu_tree_device(bh_oct_tree_gpu * h_root, bh_oct_tree_gpu ** d_root);
void allocate_kernel_params(bh_kernel_params *params);

void copy_gpu_tree_to_device(bh_oct_tree_gpu * h_root, bh_oct_tree_gpu * d_root);
void copy_gpu_tree_to_host(bh_oct_tree_gpu * h_root, bh_oct_tree_gpu * d_root);
void copy_kernel_params_to_device(bh_kernel_params params);

void free_gpu_tree_device(bh_oct_tree_gpu * d_root);
void free_kernel_params_device(bh_kernel_params params);

#endif
