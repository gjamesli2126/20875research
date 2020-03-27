/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include "bt_gpu_tree.h"


gpu_tree * gpu_transform_tree(node *root) {
	CHECK_PTR(root);

	gpu_tree *tree;
	SAFE_MALLOC(tree, sizeof(gpu_tree));

	tree->nnodes = nnodes;
	tree->depth = max_depth;

	SAFE_MALLOC(tree->nodes0, sizeof(gpu_tree_node_0)*tree->nnodes);
	SAFE_MALLOC(tree->nodes1, sizeof(gpu_tree_node_1)*tree->nnodes);

	int index = 0;
	gpu_build_tree(root, tree, &index);

	return tree;
}

static int gpu_build_tree(node *root, gpu_tree *h_tree, int* index) {
	gpu_tree_node_0 node0;
	gpu_tree_node_1 node1;

	node0.rad = root->rad;
	for (int i = 0; i < DIM; i ++) {
		node0.coord[i] = root->pivot->coord[i];
	}
	if (root->left == NULL && root->right == NULL)
		node1.idx = root->pivot->idx;
//	else
//		node1.idx = 0 - root->pre_id;
    node1.pre_id = root->pre_id;
	int my_index = *index;
	*index += 1;
	node1.depth = root->depth;

	if(root->left != NULL)
		node1.left = gpu_build_tree(root->left, h_tree, index);
	else
		node1.left = -1;

	if(root->right != NULL)
		node1.right = gpu_build_tree(root->right, h_tree, index);
	else
		node1.right = -1;

	h_tree->nodes0[my_index] =  node0;
	h_tree->nodes1[my_index] =  node1;

	return my_index;
}

gpu_tree * gpu_copy_to_dev(gpu_tree *h_tree) {
	gpu_tree * d_tree = gpu_alloc_tree_dev(h_tree);
	CUDA_SAFE_CALL(cudaMemcpy(d_tree->nodes0, h_tree->nodes0, sizeof(gpu_tree_node_0)*h_tree->nnodes, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_tree->nodes1, h_tree->nodes1, sizeof(gpu_tree_node_1)*h_tree->nnodes, cudaMemcpyHostToDevice));
	return d_tree;
}
void gpu_copy_tree_to_host(gpu_tree *d_tree, gpu_tree *h_tree) {
	// Nothing in the tree is modified to there is nothing to do here
}
void gpu_free_tree_dev(gpu_tree *d_tree) {
	CHECK_PTR(d_tree);
	CUDA_SAFE_CALL(cudaFree(d_tree->nodes0));
	CUDA_SAFE_CALL(cudaFree(d_tree->nodes1));
	CUDA_SAFE_CALL(cudaFree(d_tree->stk_node));
	CUDA_SAFE_CALL(cudaFree(d_tree->stk_axis_dist))
}
void gpu_free_tree_host(gpu_tree *h_tree) {
	CHECK_PTR(h_tree);
	free(h_tree->nodes0);
	free(h_tree->nodes1);
}
void gpu_print_tree_host(gpu_tree *h_tree) {

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
	CUDA_SAFE_CALL(cudaMalloc(&d_tree->stk_node, sizeof(int)*h_tree->depth*2*NUM_THREADS_PER_BLOCK*NUM_THREAD_BLOCKS));
	CUDA_SAFE_CALL(cudaMalloc(&d_tree->stk_axis_dist, sizeof(float)*h_tree->depth*2*NUM_THREADS_PER_BLOCK*NUM_THREAD_BLOCKS));

	return d_tree;
}

