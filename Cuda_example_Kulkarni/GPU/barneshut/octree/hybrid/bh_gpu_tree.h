#ifndef __BH_OCT_TREE_GPU_H
#define __BH_OCT_TREE_GPU_H

#include "bh.h"

typedef struct gpu_node0_ {
	vec3d cofm;
} gpu_node0;

typedef struct gpu_node1_ {
	bh_node_type type;
	float mass;
	int depth;
	int pre_id;
} gpu_node1;

typedef struct gpu_node2_ {
	int children[8];
} gpu_node2;

typedef struct gpu_node3_ {
	vec3d vel;
	vec3d acc;	
	bh_oct_tree_node *cpu_addr;
	
	#ifdef TRACK_TRAVERSALS
	int nodes_accessed;
	#endif
} gpu_node3;

typedef union {
	long long __val;
	struct {
		int index;
		float dsq;
	} items;
} stack_item;

typedef struct _bh_oct_tree_gpu {
	gpu_node0 *nodes0;
	gpu_node1 *nodes1;
	gpu_node2 *nodes2;
	gpu_node3 *nodes3;
	unsigned int nnodes;
	unsigned int depth;	

	int *stk_nodes;
	float *stk_dsq;
} bh_oct_tree_gpu;


#endif
