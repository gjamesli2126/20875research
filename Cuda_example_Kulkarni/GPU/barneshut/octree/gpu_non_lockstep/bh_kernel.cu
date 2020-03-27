/* -*- mode: c -*- */
/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/

#include <cuda.h>
#include <stdio.h>
#include "util_common.h"

#include "ptrtab.h"
#include "bh_kernel.h"

__global__ void init_kernel(void) {

}

__global__ void compute_force_gpu(bh_kernel_params params) {
	unsigned int i, pidx, j;

  int sp; 
	int tree_index; 

	float size = params.size;
  float itolsq = params.itolsq;
  int step = params.step;
  float dthf = params.dthf;
  float epssq = params.epssq;
	float nphi;
	float scale;
	float idr;
	vec3d dr;
	float drsq;
	float dsq;
	int child;

	int cur_node_index;
  
	vec3d a_prev;
	vec3d p_cofm;
	vec3d p_acc;
	vec3d p_vel;
	vec3d delta_v;
	bh_oct_tree_node * p_cpu_addr;
	
	gpu_node0 cur_node0;
	gpu_node1 cur_node1;
	gpu_node2 cur_node2;

	#ifdef TRACK_TRAVERSALS
	int nodes_accessed;
	#endif
	
	#ifndef USE_LOCAL_STACK
	int *stack_node_index;
	float *stack_dsq;
	#else
  int stack_node_index[128]; // max_depth stack
  float stack_dsq[128]; // max_depth stack
	#endif

	// loop over all points for this node
	for(i = blockIdx.x*blockDim.x + threadIdx.x; i < params.nbodies; i+= gridDim.x*blockDim.x) {
		
		// get the index of out tree node
		// possible stack collesions?
		bh_oct_tree_node * node = params.d_points_sorted[i];
		pidx = lookup_ptrtab(params.ptrtab_points_sorted, (ptrtab_key_t)node);
		
		// Cache this threads point
		//p_mass = params.root.nodes[pidx].mass;
		p_cofm = params.root.nodes0[pidx].cofm;
		p_acc = params.root.nodes3[pidx].acc;
		#ifdef TRACK_TRAVERSALS
		nodes_accessed = 0;
		#endif
		a_prev = p_acc;
		p_acc.x = 0.0;
		p_acc.y = 0.0;
		p_acc.z = 0.0;

		STACK_INIT();

	while(sp >= 1) {
		
		cur_node_index = STACK_TOP_NODE_INDEX;
		dsq = STACK_TOP_DSQ;
						
		STACK_POP();

		#ifdef TRACK_TRAVERSALS
		nodes_accessed++;
		#endif

		//CUR_NODE0 = params.root.nodes0[cur_node_index];
		dr.x = params.root.nodes0[cur_node_index].cofm.x - p_cofm.x;
		dr.y = params.root.nodes0[cur_node_index].cofm.y - p_cofm.y;
		dr.z = params.root.nodes0[cur_node_index].cofm.z - p_cofm.z;
		drsq = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z + epssq;
		 
		//params.root.nodes1[cur_node_index] = params.root.nodes1[cur_node_index];
		if(drsq < dsq) {
				if (params.root.nodes1[cur_node_index].type == bhNonLeafNode) {			
					for(j = 0; j < 8; j++) {
						child = params.root.nodes2[cur_node_index].children[j];
						if(child != -1) {		
								STACK_PUSH();
								STACK_TOP_NODE_INDEX = child;
								STACK_TOP_DSQ = dsq * 0.25;
						} else {
							break;
						}
					}

				} else {
					if(pidx != cur_node_index) {
						//drsq_2 = drsq + epssq;
						idr = rsqrtf(drsq); //1.0 / sqrt(drsq);
						nphi = params.root.nodes1[cur_node_index].mass * idr;
						scale = nphi * idr *idr;

						p_acc.x += dr.x*scale;
						p_acc.y += dr.y*scale;
						p_acc.z += dr.z*scale;
					}
				}
		} else {
			//drsq_2 = drsq + epssq;
			idr = rsqrtf(drsq); //1.0 / sqrt(drsq);
			nphi = params.root.nodes1[cur_node_index].mass * idr;
			scale = nphi * idr * idr;
						
			p_acc.x += dr.x*scale;
			p_acc.y += dr.y*scale;
			p_acc.z += dr.z*scale;
		}
	}

		//
	  p_vel = params.root.nodes3[pidx].vel;
		if(params.step > 0) {
			
		  delta_v.x = (p_acc.x - a_prev.x) * params.dthf;
		  delta_v.y = (p_acc.y - a_prev.y) * params.dthf;
		  delta_v.z = (p_acc.z - a_prev.z) * params.dthf;
		  
		  p_vel.x = p_vel.x + delta_v.x;
		  p_vel.y = p_vel.y + delta_v.y;
		  p_vel.z = p_vel.z + delta_v.z;
		}
	       
		// Write cached point back to tree
		params.root.nodes3[pidx].vel = p_vel;
		params.root.nodes3[pidx].acc = p_acc;
		#ifdef TRACK_TRAVERSALS
		params.root.nodes3[pidx].nodes_accessed = nodes_accessed;
		#endif
	}
	
}

__device__ int lookup_ptrtab(hash_table_t table, ptrtab_key_t key) {
	unsigned int index = hashkey_ptrtab(key);
	unsigned int attempt;

	for(attempt = 1; ; attempt++) {

		index = index % table.nslots;
		if(table.keys[index] == key) {
			return table.values[index];
		}

		// didn't get it so probe next slot
		index += 1;
	}
}

hash_table_t allocate_ptrtab(unsigned int nslots) {
	cudaError_t e;
	
	hash_table_t table;

	CUDA_SAFE_CALL(cudaMalloc(&(table.keys), sizeof(ptrtab_key_t) * nslots));
	CUDA_SAFE_CALL(cudaMalloc(&(table.values), sizeof(int) * nslots));
	table.nslots = nslots;

	return table;
}

void free_ptrtab(hash_table_t table) {
	CUDA_SAFE_CALL(cudaFree(table.keys));
	CUDA_SAFE_CALL(cudaFree(table.values));
}

__global__ void init_ptrtab(hash_table_t table) {
	// initialize all of the slots to zero:
	unsigned int i;
	ptrtab_key_t *key_slot;
	
	// set all slots to EMPTY_SLOT
	for(i = blockDim.x*blockIdx.x + threadIdx.x; i < table.nslots; i+=gridDim.x*blockDim.x) {
		table.keys[i] = EMPTY_KEY;
		table.values[i] = -1;
	}
}

__global__ void fill_ptrtab(hash_table_t table, bh_oct_tree_gpu d_root) {
	unsigned int i;
	ptrtab_key_t key;

	// insert each node into the poiner table
	for(i = blockDim.x*blockIdx.x + threadIdx.x; i < d_root.nnodes; i+=gridDim.x*blockDim.x) {
		
		key = (ptrtab_key_t)(d_root.nodes3[i].cpu_addr);
		insert_ptrtab(table, key, i);
	}
}

__device__ void insert_ptrtab(hash_table_t table, ptrtab_key_t key, int value) {
	unsigned int index;
	unsigned int attempt;
	ptrtab_key_t old_key;

	index = hashkey_ptrtab(key);
	ptrtab_key_t * keys;

	for(attempt = 1; ; attempt++) {
		index = index % table.nslots;
		keys = table.keys + index;
		old_key = atomicCAS((ptrtab_key_t*)keys, (ptrtab_key_t)EMPTY_KEY, (ptrtab_key_t)key);
	
		if(old_key == EMPTY_KEY && key != 0) {
			// set the value also
			table.values[index] = value;
			break;
		}
		
		// didn't get it so probe next slot
		index += 1; //attempt*attempt;
	}
}

__device__ unsigned int hashkey_ptrtab(ptrtab_key_t key) {
	unsigned int hashU = (unsigned int)(key >> 32);
	unsigned int hashL = (unsigned int)(key);

	// I'm sure this is a horrible hash function but ... whatever ...
	// I saw it mentioned in the same sentance as Don Knuth so it must work
	hashU = (hashU >> 3) * 2654435761;
	hashL = (hashL >> 3) * 2654435761;
	
	return hashU * 51 + hashL * 17;
}
