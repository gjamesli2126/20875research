/* -*- mode: c -*- */
/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/

#include <stdio.h>
#include "../../../common/util_common.h"
#include "bh_block.h"

bh_gpu_tree * build_gpu_tree(Node * root) {

	int i;
	int index;

	bh_gpu_tree * gpu_tree;
	SAFE_MALLOC(gpu_tree, sizeof(bh_gpu_tree));
	
	// get information from the cpu tree
	gpu_tree->nnodes = 0;
	gpu_tree->depth = 0;
	block_tree_info(gpu_tree, root, 1);

	// allocate the tree
	SAFE_MALLOC(gpu_tree->nodes0, sizeof(gpu_node0)*gpu_tree->nnodes);
	SAFE_MALLOC(gpu_tree->nodes1, sizeof(gpu_node1)*gpu_tree->nnodes);
	SAFE_MALLOC(gpu_tree->nodes2, sizeof(gpu_node2)*gpu_tree->nnodes);
	SAFE_MALLOC(gpu_tree->nodes3, sizeof(gpu_node3)*gpu_tree->nnodes);

	index = 0;
	block_gpu_tree(root, gpu_tree, &index);
 
	return gpu_tree;
}

int block_gpu_tree(Node * node, bh_gpu_tree * root, int * index) {

	int i;
	int my_index = -1;

	// Save the current index as ours and go to next free position
	my_index = *index;
	*index = *index + 1; 

	// copy the node data
	gpu_node0 *gnode0 = &(root->nodes0[my_index]);
	gpu_node1 *gnode1 = &(root->nodes1[my_index]);
	gpu_node3 *gnode3 = &(root->nodes3[my_index]);

	gnode1->ropen = node->ropen;
	gnode1->mass = node->mass;
	gnode0->cofm = node->cofm;
	gnode0->point_id = node->point_id;
	gnode0->leafNode = node->leafNode;
	gnode3->pre_id = node->pre_id;
	gnode3->depth = node->depth;
	#ifdef TRACK_TRAVERSALS
	gnode3->nodes_accessed = 0;
	#endif
//	gnode3->cpu_addr = node;

	if(node->left != NULL) {
		root->nodes2[my_index].left = block_gpu_tree(node->left, root, index);
	}
	if(node->right != NULL) {
		root->nodes2[my_index].right = block_gpu_tree(node->right, root, index);
	}

	// return the node index
	return my_index;
}

void block_tree_info(bh_gpu_tree * gpu_root, Node * root, int depth) {
	int i;

  // update maximum depth
	if(depth > gpu_root->depth)
		gpu_root->depth = depth;

  
	// update number of nodes
	gpu_root->nnodes++;

	// goto children
	if(root->left != NULL) {
		block_tree_info(gpu_root, root->left, depth+1);
	}
	if(root->right != NULL) {
		block_tree_info(gpu_root, root->right, depth+1);
	}
}

void free_gpu_tree(bh_gpu_tree * root) {
	free(root->nodes0);
	free(root->nodes1);
	free(root->nodes2);
	free(root->nodes3);
}



