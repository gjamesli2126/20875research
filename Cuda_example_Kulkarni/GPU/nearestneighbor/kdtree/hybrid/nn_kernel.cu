/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include "nn_kernel.h"

#define STACK_INIT()	sp = 1;
#define STACK_PUSH(node) sp = sp + 1; stk[WARP_INDEX][sp] = node
#define STACK_POP() sp = sp - 1;
#define STACK_NODE stk[WARP_INDEX][sp]

__global__ void init_kernel(void) {

}
__global__ void nearest_neighbor_search(kernel_params params)
{
 	float search_points_coord[DIM];
	int closest;
	float closestDist;

#ifdef TRACK_TRAVERSALS
	int num_nodes_traversed;
#endif

    gpu_tree d_tree = params.d_tree;
    gpu_point *d_training_points = params.d_training_points;
    gpu_point *d_search_points = params.d_search_points;
    int n_search_points = params.n_search_points;
    int *d_array_points = params.d_array_points;

	int i, j;
	
	int cur_node_index;
	__shared__ int SP[NUM_OF_WARPS_PER_BLOCK];
#define sp SP[WARP_INDEX]

	bool cond, status;
    bool opt1, opt2;
	int critical;
	unsigned int vote_left;
	unsigned int vote_right;
	unsigned int num_left;
	unsigned int num_right;
	
	__shared__ int stk[NUM_OF_WARPS_PER_BLOCK][64];	// original value is 64

	gpu_tree_node_0 cur_node0;
	gpu_tree_node_2 cur_node2;
	gpu_tree_node_3 cur_node3;

	float dist=0.0;
	float boxdist=0.0;
	float sum=0.0;
	float boxsum=0.0;
	float center=0.0;
	int id = 0;
	
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n_search_points; idx += blockDim.x * gridDim.x)
    {
		int pidx = d_array_points[idx];
        for(j = 0; j < DIM; j++)
        {
            search_points_coord[j] = d_search_points[pidx].coord[j];
        }
		
        closest = d_search_points[pidx].closest;
        closestDist = d_search_points[pidx].closest_dist;
        #ifdef TRACK_TRAVERSALS
        num_nodes_traversed = d_search_points[pidx].num_nodes_traversed;
        #endif

        STACK_INIT ();
		STACK_NODE = 1;
		status = 1;
		critical = 63;
		cond = 1;
        while(sp >= 1)
        {
			cur_node_index = STACK_NODE;

#ifdef TRACK_TRAVERSALS
                num_nodes_traversed++;
#endif

			if (status == 0 && critical >= sp) {
				status = 1;
			}
            
            STACK_POP();

//            if (pidx == 0 && status == 0) {
//                printf("YES!\n");
//            }


			if (status == 1) {
/*				static int id = 0;
				if (pidx == 0) {
					printf("%dth visit!\n", id ++);
				}*/

            // inlined function can_correlate
				dist=0.0;
				boxdist=0.0;
				sum=0.0;
				boxsum=0.0;
				center=0.0;

				for(i = 0; i < DIM; i++)
				{
					float max = d_tree.nodes1[cur_node_index].items[i].max;
					float min = d_tree.nodes1[cur_node_index].items[i].min;
					center = (max + min) / 2;
					boxdist = (max - min) / 2;
					dist = search_points_coord[i] - center;
					sum += dist * dist;
					boxsum += boxdist * boxdist;
				}
//                critical = sp ;
				cond = (sqrt(sum) - sqrt(boxsum) < sqrt(closestDist));
//			}

			if(!__any(cond)) {
				continue;
			}

//			if (status == 1) {
            critical = sp;
				if (!cond) {
					status = 0;
//					critical = sp;
				} else {
/*					if (pidx == 0) {
						printf("%dth visit @ %d!\n", id ++, cur_node_index);
					}*/
					cur_node0 = d_tree.nodes0[cur_node_index];
					if(cur_node0.items.axis == DIM) {
						cur_node3 = d_tree.nodes3[cur_node_index];
						for(i = 0; i < MAX_POINTS_IN_CELL; i++) {
							if(cur_node3.items.points[i] >= 0) {
								// update closest...
								float dist = 0.0;
								float t;
								for(j = 0; j < DIM; j++) {
									t = (d_training_points[cur_node3.items.points[i]].coord[j] - search_points_coord[j]);
									dist += t*t;
								}

								if(dist <= closestDist) {
									closest = cur_node3.items.points[i];
									closestDist = dist;
								}
							}

						}
					} else {
						cur_node2 = d_tree.nodes2[cur_node_index];
						opt1 = search_points_coord[cur_node0.items.axis] < cur_node0.items.splitval;
						opt2 = search_points_coord[cur_node0.items.axis] >= cur_node0.items.splitval;
						vote_left = __ballot(opt1);
						vote_right = __ballot(opt2);
						num_left = __popc(vote_left);
						num_right = __popc(vote_right);
						// majority vote
						if (num_left > num_right) {
							if(RIGHT != NULL_NODE) { STACK_PUSH(RIGHT); }
							if(LEFT != NULL_NODE) { STACK_PUSH(LEFT); }
						} else {
							if(LEFT != NULL_NODE) { STACK_PUSH(LEFT); }
							if(RIGHT != NULL_NODE) { STACK_PUSH(RIGHT); }
						}
					}
				}
			}
        }

        d_search_points[pidx].closest = closest;
        d_search_points[pidx].closest_dist = closestDist;
        #ifdef TRACK_TRAVERSALS
        d_search_points[pidx].num_nodes_traversed = num_nodes_traversed;
        #endif
    }
}

////////////////////////////////////////////////////
////////////////////////////////////////////////////
////////////////////////////////////////////////////

static void gpu_alloc_tree_host(gpu_tree * h_tree);
static void gpu_init_tree_properties(gpu_tree *h_tree, KDCell *root, int depth);
static int gpu_build_tree(KDCell *root, gpu_tree *h_tree, int index, int depth, int parent_index);

gpu_tree * gpu_transform_tree(KDCell *root) 
{

	CHECK_PTR(root);
	
	gpu_tree *tree;
	SAFE_MALLOC(tree, sizeof(gpu_tree));

	tree->nnodes = 0;
	tree->depth = 0;

	gpu_init_tree_properties(tree, root, 1);
	gpu_alloc_tree_host(tree);
	
	int index = 1;
	gpu_build_tree(root, tree, index, 0, NULL_NODE);

	return tree;
}

void gpu_free_tree_host(gpu_tree *h_tree) 
{
	CHECK_PTR(h_tree);
	free(h_tree->nodes0);	
	free(h_tree->nodes1);
    free(h_tree->nodes2);
	free(h_tree->nodes3);
}

static void gpu_alloc_tree_host(gpu_tree * h_tree) 
{
	int n = h_tree->depth;
	h_tree->max_nnodes = 1;
	while(n > 0)
	{
		h_tree->max_nnodes *= 2;
		n --;
	}

//	printf("the number of nodes is %d, and the number of the max nodes is %d.\n", h_tree->nnodes, h_tree->max_nnodes);

	SAFE_MALLOC(h_tree->nodes0, sizeof(gpu_tree_node_0)*h_tree->max_nnodes);
	SAFE_MALLOC(h_tree->nodes1, sizeof(gpu_tree_node_1)*h_tree->max_nnodes);
	SAFE_MALLOC(h_tree->nodes2, sizeof(gpu_tree_node_2)*h_tree->max_nnodes);
	SAFE_MALLOC(h_tree->nodes3, sizeof(gpu_tree_node_3)*h_tree->max_nnodes);
}

static void gpu_init_tree_properties(gpu_tree * h_tree, KDCell * root, int depth) 
{

	h_tree->nnodes++;

	if(depth > h_tree->depth) 
		h_tree->depth = depth;

	if(root->left != NULL)
		gpu_init_tree_properties(h_tree, root->left, depth + 1);

	if(root->right != NULL)
		gpu_init_tree_properties(h_tree, root->right, depth + 1);
}

static int gpu_build_tree(KDCell *root, gpu_tree *h_tree, int index, int depth, int parent_index)
{
	// add node to tree
	gpu_tree_node_0 node0;
	gpu_tree_node_1 node1;
	gpu_tree_node_2 node2;
	gpu_tree_node_3 node3;
	int i;
	int my_index = index;

	node0.items.axis = root->axis;
	node0.items.splitval = root->splitval;
	node0.items.depth = root->depth;
	node0.items.pre_id = root->pre_id;
	for(i = 0; i < DIM; i++) {
		node1.items[i].min = root->min[i];
		node1.items[i].max = root->max[i];
	}

	for(i = 0; i < MAX_POINTS_IN_CELL; i++) {
		node3.items.points[i] = root->points[i];
	}

	//node1.parent = parent_index;
	if(root->left != NULL)
		node2.items.left = gpu_build_tree(root->left, h_tree, 2*index, depth + 1, my_index);
	else
		node2.items.left = NULL_NODE;
	
	if(root->right != NULL) {
		node2.items.right = gpu_build_tree(root->right, h_tree, 2*index+1, depth + 1, my_index);
	} else {
		node2.items.right = NULL_NODE;
	}
	
	h_tree->nodes0[my_index] =  node0;
	h_tree->nodes1[my_index] =  node1;
	h_tree->nodes2[my_index] =  node2;
	h_tree->nodes3[my_index] =  node3;
	return my_index;
}





