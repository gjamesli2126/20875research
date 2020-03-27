/* -*- mode: c -*- */
/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/

#include <stdio.h>
#include "../../../common/util_common.h"
#include "bh_block.h"

bh_oct_tree_gpu * build_gpu_tree(bh_oct_tree_node * root) 
{
	int i;
	int index;

	bh_oct_tree_gpu * gpu_tree;
	SAFE_MALLOC(gpu_tree, sizeof(bh_oct_tree_gpu));
	
	// get information from the cpu tree
	gpu_tree->nnodes = 0;
	gpu_tree->depth = 0;
	block_tree_info(gpu_tree, root, 1);

	printf("the depth of the tree is: %d, and the number of nodes is: %d.\n", gpu_tree->depth, gpu_tree->nnodes);

	// allocate the tree
	SAFE_MALLOC(gpu_tree->nodes0, sizeof(gpu_node0)*gpu_tree->nnodes);
	SAFE_MALLOC(gpu_tree->nodes1, sizeof(gpu_node1)*gpu_tree->nnodes);
	SAFE_MALLOC(gpu_tree->nodes2, sizeof(gpu_node2)*gpu_tree->nnodes);
	SAFE_MALLOC(gpu_tree->nodes3, sizeof(gpu_node3)*gpu_tree->nnodes);

	index = 0;
	block_gpu_tree(root, gpu_tree, &index, BLOCK_SIZE, 0);
 
	return gpu_tree;
}

int block_gpu_tree(bh_oct_tree_node * node, bh_oct_tree_gpu * root, int * index, int max_block_size, int depth) 
{
	int i;
	int child_id;
	int my_index = -1;

	// Save the current index as ours and go to next free position
	my_index = *index;
	*index = *index + 1; 

	// copy the node data
	gpu_node0 *gnode0 = &(root->nodes0[my_index]);
	gpu_node1 *gnode1 = &(root->nodes1[my_index]);
	gpu_node3 *gnode3 = &(root->nodes3[my_index]);

	gnode1->type = node->type;
	gnode1->mass = node->mass;
	gnode1->depth = node->depth;
	gnode1->pre_id = node->pre_id;
	gnode0->cofm = node->cofm;
	gnode3->vel = node->vel;
	gnode3->acc = node->acc;
	gnode3->cpu_addr = node;

	for(i = 0; i < 8; i++) {
		if(node->children[i] != 0) {
			child_id = block_gpu_tree(node->children[i], root, index, max_block_size, depth+1);
			root->nodes2[my_index].children[i] = child_id;
		} else {
			root->nodes2[my_index].children[i] = -1;
		}
	}

	// return the node index
	return my_index;
}

void block_tree_info(bh_oct_tree_gpu * gpu_root, bh_oct_tree_node * root, int depth) 
{
	int i;

  // update maximum depth
	if(depth > gpu_root->depth)
		gpu_root->depth = depth;

  
	// update number of nodes
	gpu_root->nnodes++;

	// goto children
	for(i=0; i < 8; i++) {
		if(root->children[i] != NULL)
			block_tree_info(gpu_root, root->children[i], depth+1);
	}
}

void free_gpu_tree(bh_oct_tree_gpu * root) {
	free(root->nodes0);
	free(root->nodes1);
	free(root->nodes2);
	free(root->nodes3);
}



