/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include "pc_kernel.h"

gpu_tree * build_gpu_tree(kd_cell * c_root)
{
    int index = 1;
    gpu_tree *h_tree;
    SAFE_MALLOC(h_tree, sizeof(gpu_tree));
    
    // get the information from the cpu tree
    h_tree->nnodes = 0;
    h_tree->tree_depth = 0;
    block_tree_info(h_tree, c_root, 1);
    
    int n = h_tree->tree_depth;
    fprintf(stdout,"@ tree_height: %d\n", n);
    h_tree->max_nnodes = 1;
    while (n > 0)
    {
        h_tree->max_nnodes *= 2;
        n --;
    }
    printf("The max number of the nodes is %d.\n", h_tree->max_nnodes);

    // allocate the tree
    SAFE_MALLOC(h_tree->nodes0, sizeof(gpu_node0) * h_tree->max_nnodes);
    SAFE_MALLOC(h_tree->nodes1, sizeof(gpu_node1) * h_tree->max_nnodes);
    SAFE_MALLOC(h_tree->nodes2, sizeof(gpu_node2) * h_tree->max_nnodes);
    SAFE_MALLOC(h_tree->nodes3, sizeof(gpu_node3) * h_tree->max_nnodes);
    
    block_gpu_tree(c_root, h_tree, index, 0);
    
    return h_tree;
}

void block_tree_info(gpu_tree * h_root, kd_cell * c_root, int depth)
{
    //update the maximum depth
    if (depth > h_root->tree_depth)
    {
        h_root->tree_depth = depth;
    }
    
    // update the number of nodes
    h_root->nnodes ++;
    
    // goto the children
    if (c_root->left != NULL)
    {
        block_tree_info(h_root, c_root->left, depth + 1);
    }
    if (c_root->right != NULL)
    {
        block_tree_info(h_root, c_root->right, depth + 1);
    }
}

int block_gpu_tree(kd_cell * c_node, gpu_tree * h_root, int index, int depth)
{
    int i = 0;
    int my_index = -1;
    
    //Save the current index as ours and go to next position
    my_index = index;
//    index = index + 1;
    
    //copy the node data
    h_root->nodes3[my_index].corr = 0;
    h_root->nodes1[my_index].splitType = c_node->splitType;
    h_root->nodes1[my_index].depth = c_node->depth;
    h_root->nodes1[my_index].pre_id = c_node->pre_id;
    for (i = 0; i < DIM; i ++)
    {
        h_root->nodes0[my_index].coord[i].items.max = c_node->coord_max[i];
        h_root->nodes0[my_index].coord[i].items.min = c_node->min[i];
    }

//    h_root->nodes3[my_index].cpu_addr = c_node;
	h_root->nodes3[my_index].point_id = -1;
//    printf("the index of gpu node is: %d, and the index of the cpu node is %d.\n", my_index, c_node->id);

    if (c_node->left != NULL)
    {
        h_root->nodes2[my_index].left = block_gpu_tree(c_node->left, h_root, 2 * index, depth + 1);
    }
    else
    {
        h_root->nodes2[my_index].left = -1;
    }
    
    if (c_node->right != NULL)
    {
        h_root->nodes2[my_index].right = block_gpu_tree(c_node->right, h_root, 2 * index + 1, depth + 1);
    }
    else
    {
        h_root->nodes2[my_index].right = -1;
    }
    
//#ifdef TRACK_TRAVERSALS
//    h_root->nodes0[my_index].nodes_accessed ++;
//#endif
    
    // return the node index
    return my_index;
}

void free_gpu_tree(gpu_tree * h_root)
{
    free(h_root->nodes0);
    free(h_root->nodes1);
    free(h_root->nodes2);
    free(h_root->nodes3);
    free(h_root);
}

__global__ void init_kernel(void)
{
    
}

__global__ void compute_correlation(pc_kernel_params params)
{
#ifdef USE_SMEM
    __shared__ float p_coord[DIM][NUM_OF_THREADS_PER_BLOCK];
	__shared__ gpu_node0 cur_node0[NUM_OF_WARPS_PER_BLOCK];
	__shared__ gpu_node1 cur_node1[NUM_OF_WARPS_PER_BLOCK];
	__shared__ gpu_node2 cur_node2[NUM_OF_WARPS_PER_BLOCK];
    __shared__ int stack[NWARPS_PER_BLOCK][128];
    //__shared__ int mask[NWARPS_PER_BLOCK][128];
#else
	float p_coord[DIM];
	gpu_node0 cur_node0;
	gpu_node1 cur_node1;
	gpu_node2 cur_node2;
//    int stack[128];
	//unsigned int mask[128];
#endif
    
    int cur_node_index = 0;
    int i = 0;
    int j = 0;
    int can_correlate_result = 0;
    int p_corr = 0;
    //unsigned int mask[128];
    
    float rad = 0.0;
    float dist = 0.0;
    float sum = 0.0;
    float boxsum = 0.0;
    float boxdist = 0.0;
    float center = 0.0;
    
	bool cond, status;
    bool opt1, opt2;
	unsigned int critical;
	unsigned int vote_left;
	unsigned int vote_right;
	unsigned int num_left;
	unsigned int num_right;
	__shared__ int stack[NUM_OF_WARPS_PER_BLOCK][64];
	__shared__ unsigned int SP[NUM_OF_WARPS_PER_BLOCK];
#define sp SP[WARP_INDEX]

#define STACK_TOP_NODE_INDEX stack[WARP_INDEX][sp]
#define STACK_INIT() sp = 1; stack[WARP_INDEX][1] = 0

#ifdef TRACK_TRAVERSALS
	int p_nodes_accessed;
	int p_nodes_truncated;
#endif
    
    rad = params.rad;
//    for (i = blockIdx.x*blockDim.x+threadIdx.x; i < params.npoints; i += gridDim.x*blockDim.x)
    for (int pidx = blockIdx.x*blockDim.x+threadIdx.x; pidx < params.npoints; pidx += gridDim.x*blockDim.x)
    {
		i = params.index_buffer[pidx];
        p_corr = params.set.nodes3[i].corr;
#ifdef TRACK_TRAVERSALS
        p_nodes_accessed = 0;
#endif
        for (j = 0; j < DIM; j ++)
        {
            p_coord[j] = params.set.nodes0[i].coord[j].items.max;
        }
        
        STACK_INIT();
		STACK_TOP_NODE_INDEX = params.root_index;
		status = 1;
		critical = 63;
		cond = 1;
        
        while (sp >= 1)
        {
            cur_node_index = STACK_TOP_NODE_INDEX;
            //cur_mask = STACK_TOP_MASK;

			if (status == 0 && critical >= sp) {
				status = 1;
			}
            
            STACK_POP();
           
#ifdef TRACK_TRAVERSALS
            p_nodes_accessed ++; 
#endif
 
            if (status) {
                CUR_NODE0 = params.tree.nodes0[cur_node_index];
            	// inline call: can_correlate(...)
				critical = sp;
            	sum = 0.0;
            	boxsum = 0.0;
            	for (j = 0; j < DIM; j ++)
            	{
            		center = (CUR_NODE0.coord[j].items.max + CUR_NODE0.coord[j].items.min) / 2;
            		boxdist = (CUR_NODE0.coord[j].items.max - CUR_NODE0.coord[j].items.min) / 2;
            		dist = p_coord[j] - center;
            		sum += dist * dist;
            		boxsum += boxdist * boxdist;
            	}

            	cond = sqrt(sum) - sqrt(boxsum) < rad;
            
            	if (!__any(cond))
            	{
#ifdef TRACK_TRAVERSALS
                	p_nodes_truncated ++;
#endif
            		continue;
            	}

//                critical = sp;
            	if (!cond) {
					status = 0;
//					critical = sp;
            	} else {
            		CUR_NODE1 = params.tree.nodes1[cur_node_index];
            		if (CUR_NODE1.splitType == SPLIT_LEAF)
            		{
            			// inline call: in_radii(...)
            			dist = 0.0;
            			for (j = 0; j < DIM; j ++)
            			{
            				dist += (p_coord[j] - CUR_NODE0.coord[j].items.max) * (p_coord[j] - CUR_NODE0.coord[j].items.max);
            			}
                    
            			dist = sqrt(dist);
            			if (dist < rad)
            			{
            				p_corr ++; // =(100 * block_node_coord[0][WARP_INDEX][k]);
            			}
            		} else {
            			CUR_NODE2 = params.tree.nodes2[cur_node_index];
            			// push children
            			if (CUR_NODE2.right != -1)
            			{
            				STACK_PUSH();
            				STACK_TOP_NODE_INDEX = CUR_NODE2.right;
            				// STACK_TOP_MASK = __ballot(can_correlate_result);
            			}

            			if (CUR_NODE2.left != -1)
            			{
            				STACK_PUSH();
            				STACK_TOP_NODE_INDEX = CUR_NODE2.left;
            				// STACK_TOP_MASK = __ballot(can_correlate_result);
            			}
            		}
            	}
            }
        }
	           
        params.set.nodes3[i].corr = p_corr;
#ifdef TRACK_TRAVERSALS
        params.set.nodes0[i].nodes_accessed = p_nodes_accessed;
#endif
    }
#undef sp
#define STACK_INIT() sp = 1; stack[1] = 0
#define STACK_TOP_NODE_INDEX stack[sp]
}
