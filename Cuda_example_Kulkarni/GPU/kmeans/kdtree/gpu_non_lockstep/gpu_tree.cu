/* -*- mode: c -*- */
/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include <stdio.h>
#include <stdlib.h>

#include "../../../common/util_common.h"
#include "common.h"
#include "gpu_tree.h"

gpu_tree * gpu_transform_tree(Node *root) {

	CHECK_PTR(root);	
	gpu_tree *tree;
	SAFE_MALLOC(tree, sizeof(gpu_tree));

	tree->nnodes = 0;
	tree->depth = 0;

	gpu_init_tree_properties(tree, root, 0);
	gpu_alloc_tree_host(tree);
	
	int index = 0;
	gpu_build_tree(root, tree, &index, 0, -1);

	return tree;
}

void gpu_free_tree_host(gpu_tree *h_tree) {
	CHECK_PTR(h_tree);
	free(h_tree->nodes0);	
	free(h_tree->nodes1);
  	free(h_tree->nodes2);
	free(h_tree->nodes3);
}

static void gpu_alloc_tree_host(gpu_tree * h_tree) {
	SAFE_MALLOC(h_tree->nodes0, sizeof(gpu_tree_node_0)*h_tree->nnodes);
	SAFE_MALLOC(h_tree->nodes1, sizeof(gpu_tree_node_1)*h_tree->nnodes);
	SAFE_MALLOC(h_tree->nodes2, sizeof(gpu_tree_node_2)*h_tree->nnodes);
	SAFE_MALLOC(h_tree->nodes3, sizeof(gpu_tree_node_3)*h_tree->nnodes);
}

static void gpu_init_tree_properties(gpu_tree * h_tree, Node * root, int depth) {
	h_tree->nnodes++;
	if(depth > h_tree->depth) 
		h_tree->depth = depth;

	if(root->left != NULL)
		gpu_init_tree_properties(h_tree, root->left, depth + 1);
	if(root->right != NULL)
		gpu_init_tree_properties(h_tree, root->right, depth + 1);
}

static int gpu_build_tree(Node *root, gpu_tree *h_tree, int *index, int depth, int parent_index) {
	// add node to tree
	gpu_tree_node_0 node0;
	gpu_tree_node_1 node1;
	gpu_tree_node_2 node2;
	gpu_tree_node_3 node3;
	int i;
	int my_index = *index; 
	*index += 1;

	node0.axis = root->axis;
	node0.depth = root->depth;
	node0.clusterId = root->pivot->pt.clusterId;
	node0.parent = parent_index;
	for(i = 0; i < DIM; i++) {
		node1.coord[i] = root->pivot->pt.coord[i];
	}

	//node1.parent = parent_index;
	if(root->left != NULL)
		node2.left = gpu_build_tree(root->left, h_tree, index, depth + 1, my_index);
	else
		node2.left = -1;
	
	if(root->right != NULL) {
		node2.right = gpu_build_tree(root->right, h_tree, index, depth + 1, my_index);
	} else {
		node2.right = -1;
	}
	
	h_tree->nodes0[my_index] =  node0;
	h_tree->nodes1[my_index] =  node1;
	h_tree->nodes2[my_index] =  node2;
	h_tree->nodes3[my_index] =  node3;
	return my_index;
}

gpu_tree * gpu_copy_to_dev(gpu_tree *h_tree) {

	gpu_tree * d_tree = gpu_alloc_tree_dev(h_tree);
	
	CUDA_SAFE_CALL(cudaMemcpy(d_tree->nodes0, h_tree->nodes0, sizeof(gpu_tree_node_0)*h_tree->nnodes, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_tree->nodes1, h_tree->nodes1, sizeof(gpu_tree_node_1)*h_tree->nnodes, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_tree->nodes2, h_tree->nodes2, sizeof(gpu_tree_node_2)*h_tree->nnodes, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_tree->nodes3, h_tree->nodes3, sizeof(gpu_tree_node_3)*h_tree->nnodes, cudaMemcpyHostToDevice));
	
	return d_tree;
}

void gpu_copy_tree_to_host(gpu_tree *d_tree, gpu_tree *h_tree) {
	// Nothing in the tree is modified to there is nothing to do here!
}

void gpu_free_tree_dev(gpu_tree *d_tree) {
	CHECK_PTR(d_tree);
	CUDA_SAFE_CALL(cudaFree(d_tree->nodes0));
	CUDA_SAFE_CALL(cudaFree(d_tree->nodes1));
	CUDA_SAFE_CALL(cudaFree(d_tree->nodes2));
	CUDA_SAFE_CALL(cudaFree(d_tree->nodes3));
	CUDA_SAFE_CALL(cudaFree(d_tree->visited));
}

static gpu_tree* gpu_alloc_tree_dev(gpu_tree *h_tree) {
	
	CHECK_PTR(h_tree);
	
	gpu_tree * d_tree;
	SAFE_MALLOC(d_tree, sizeof(gpu_tree));
	
	// copy tree value params:
	d_tree->nnodes = h_tree->nnodes;
	d_tree->depth = h_tree->depth;

	CUDA_SAFE_CALL(cudaMalloc(&d_tree->nodes0, sizeof(gpu_tree_node_0)*h_tree->nnodes));
	CUDA_SAFE_CALL(cudaMalloc(&d_tree->nodes1, sizeof(gpu_tree_node_1)*h_tree->nnodes));	
	CUDA_SAFE_CALL(cudaMalloc(&d_tree->nodes2, sizeof(gpu_tree_node_2)*h_tree->nnodes));
	CUDA_SAFE_CALL(cudaMalloc(&d_tree->nodes3, sizeof(gpu_tree_node_3)*h_tree->nnodes));
	CUDA_SAFE_CALL(cudaMalloc(&d_tree->visited, sizeof(bool)*npoints*K));

	return d_tree;
}
