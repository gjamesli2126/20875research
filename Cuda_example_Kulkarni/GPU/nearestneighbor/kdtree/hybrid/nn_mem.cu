/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include "nn_mem.h"

static gpu_point* gpu_alloc_points_host(unsigned int npoints);
static gpu_point *gpu_alloc_points_dev(unsigned int npoints);

gpu_tree* alloc_tree_dev(gpu_tree *h_tree)
{
	CHECK_PTR(h_tree);
	
	gpu_tree * d_tree;
	SAFE_MALLOC(d_tree, sizeof(gpu_tree));
	
	// copy tree value params:
	d_tree->nnodes = h_tree->nnodes;
	d_tree->max_nnodes = h_tree->max_nnodes;
	d_tree->depth = h_tree->depth;
    printf("the number of nodes is %d, the depth is %d\n", h_tree->max_nnodes, h_tree->depth);

	CUDA_SAFE_CALL(cudaMalloc(&(d_tree->nodes0), sizeof(gpu_tree_node_0)*h_tree->max_nnodes));
	CUDA_SAFE_CALL(cudaMalloc(&(d_tree->nodes1), sizeof(gpu_tree_node_1)*h_tree->max_nnodes));
	CUDA_SAFE_CALL(cudaMalloc(&(d_tree->nodes2), sizeof(gpu_tree_node_2)*h_tree->max_nnodes));
	CUDA_SAFE_CALL(cudaMalloc(&(d_tree->nodes3), sizeof(gpu_tree_node_3)*h_tree->max_nnodes));
	CUDA_SAFE_CALL(cudaMalloc(&d_tree->stk, sizeof(int)*h_tree->depth*NUM_OF_THREADS_PER_BLOCK*NUM_OF_BLOCKS));

	return d_tree;
}

gpu_tree* copy_tree_to_dev(gpu_tree *h_tree)
{
	gpu_tree* d_tree = alloc_tree_dev(h_tree);

	CUDA_SAFE_CALL(cudaMemcpy(d_tree->nodes0, h_tree->nodes0, sizeof(gpu_tree_node_0)*h_tree->max_nnodes, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_tree->nodes1, h_tree->nodes1, sizeof(gpu_tree_node_1)*h_tree->max_nnodes, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_tree->nodes2, h_tree->nodes2, sizeof(gpu_tree_node_2)*h_tree->max_nnodes, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_tree->nodes3, h_tree->nodes3, sizeof(gpu_tree_node_3)*h_tree->max_nnodes, cudaMemcpyHostToDevice));

	return d_tree;
}

void free_tree_dev(gpu_tree *d_tree)
{
	CHECK_PTR(d_tree);
	CUDA_SAFE_CALL(cudaFree(d_tree->nodes0));
	CUDA_SAFE_CALL(cudaFree(d_tree->nodes1));
	CUDA_SAFE_CALL(cudaFree(d_tree->nodes2));
	CUDA_SAFE_CALL(cudaFree(d_tree->nodes3));
	CUDA_SAFE_CALL(cudaFree(d_tree->stk));
	d_tree = NULL;
}


static gpu_point* gpu_alloc_points_host(unsigned int npoints) 
{
	gpu_point *p;
	SAFE_MALLOC(p, sizeof(gpu_point)*npoints);
	return p;
}

static gpu_point *gpu_alloc_points_dev(unsigned int npoints) 
{
	gpu_point *d_points;
	CUDA_SAFE_CALL(cudaMalloc(&d_points, sizeof(gpu_point)*npoints));
	return d_points;
}

gpu_point * gpu_transform_points(Point *points, unsigned int npoints) 
{
	int i, j;
	gpu_point *p = gpu_alloc_points_host(npoints);
	CHECK_PTR(p);
	
//	printf("what's up, man?\n");
	for(i = 0; i < npoints; i++) 
	{
		p[i].closest = points[i].closest;
		p[i].closest_dist = points[i].closest_dist;
        p[i].id = points[i].id;
		#ifdef TRACK_TRAVERSALS
		p[i].num_nodes_traversed = points[i].num_nodes_traversed;
		#endif
		for(j = 0; j < DIM; j++)
		{
			p[i].coord[j] = points[i].coord[j];
		}
	}

	return p;
}

void gpu_free_points_host(gpu_point *h_points) 
{
	free(h_points);
	h_points = NULL;
}

void gpu_free_points_dev(gpu_point *d_points) 
{
	CUDA_SAFE_CALL(cudaFree(d_points));
	d_points = NULL;
}

gpu_point *gpu_copy_points_to_dev(gpu_point *h_points, unsigned int npoints) 
{
	gpu_point *d_points = gpu_alloc_points_dev(npoints);
	CUDA_SAFE_CALL(cudaMemcpy(d_points, h_points, sizeof(gpu_point)*npoints, cudaMemcpyHostToDevice));
	return d_points;
}

/*void gpu_copy_points_to_host(gpu_point *d_points, gpu_point *h_points, SpliceNode *sn, unsigned int npoints)
{
	int i;
	CUDA_SAFE_CALL(cudaMemcpy(h_points, d_points, sizeof(gpu_point)*npoints, cudaMemcpyDeviceToHost));
	for(i = 0; i < npoints; i++) 
	{
		sn->points[i]->closest = h_points[i].closest;
		sn->points[i]->closest_dist = h_points[i].closest_dist;
		#ifdef TRACK_TRAVERSALS
		sn->points[i]->num_nodes_traversed = h_points[i].num_nodes_traversed;
		#endif
	}
}*/

void gpu_copy_points_to_host(gpu_point *d_points, gpu_point *h_points, Point *points, unsigned int npoints) {
    int i;
    CUDA_SAFE_CALL(cudaMemcpy(h_points, d_points, sizeof(gpu_point)*npoints, cudaMemcpyDeviceToHost));
    for(i = 0; i < npoints; i++) {
        points[i].closest = h_points[i].closest;
        points[i].closest_dist = h_points[i].closest_dist;
        #ifdef TRACK_TRAVERSALS
        points[i].num_nodes_traversed = h_points[i].num_nodes_traversed;
        #endif
    }
}

void alloc_set_dev(gpu_point_set *h_set, gpu_point_set *d_set)
{
	CUDA_SAFE_CALL(cudaMalloc(&(d_set->points), sizeof(gpu_point)*h_set->npoints));
}

void copy_set_to_dev(gpu_point_set *h_set, gpu_point_set *d_set)
{
	CUDA_SAFE_CALL(cudaMemcpy(d_set->points, h_set->points, sizeof(gpu_point)*h_set->npoints, cudaMemcpyHostToDevice));
}

void copy_set_to_host(gpu_point_set *h_set, gpu_point_set *d_set)
{
	CUDA_SAFE_CALL(cudaMemcpy(h_set->points, d_set->points, sizeof(gpu_point)*h_set->npoints, cudaMemcpyDeviceToHost));
}

void free_set_dev(gpu_point_set *d_set)
{
	CUDA_SAFE_CALL(cudaFree(d_set->points));
}

