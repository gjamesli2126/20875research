#ifndef __PTRTAB_H_
#define __PTRTAB_H_

#include "bh_gpu_tree.h"

#define EMPTY_KEY (ptrtab_key_t)((~0))

typedef unsigned long long int ptrtab_key_t;

typedef struct _hash_table_t {
  ptrtab_key_t *keys;
	int *values;
	unsigned int nslots;
} hash_table_t;

hash_table_t allocate_ptrtab(unsigned int nslots);
void free_ptrtab(hash_table_t table);

__global__ void init_ptrtab(hash_table_t table);
__global__ void fill_ptrtab(hash_table_t table,  bh_oct_tree_gpu d_root);

__device__ void insert_ptrtab(hash_table_t table, ptrtab_key_t key, int value);
__device__ int lookup_ptrtab(hash_table_t table, ptrtab_key_t key);
__device__ unsigned int hashkey_ptrtab(ptrtab_key_t key);

#endif
