/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include "pc_pre_kernel.h"

__global__ void pre_compute_correlation(pc_pre_kernel_params params, int start, int end)
{
#ifdef USE_SMEM
    __shared__ float p_coord[DIM][THREADS_PER_BLOCK];
    __shared__ gpu_node0 cur_node0[NWARPS_PER_BLOCK];
    __shared__ gpu_node1 cur_node1[NWARPS_PER_BLOCK];
    __shared__ gpu_node2 cur_node2[NWARPS_PER_BLOCK];
    __shared__ int stack[NWARPS_PER_BLOCK][64];
    //__shared__ int mask[NWARPS_PER_BLOCK][128];
#else
    float p_coord[DIM];
    gpu_node0 cur_node0;
    gpu_node1 cur_node1;
    gpu_node2 cur_node2;
    int stack[128];
    //unsigned int mask[128];
#endif

    int sp = 0;
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

#ifdef TRACK_TRAVERSALS
    int p_nodes_accessed = 0;
    int p_nodes_truncated = 0;
#endif

    rad = params.rad;
    for (i = blockIdx.x*blockDim.x+threadIdx.x + start; i < end; i += gridDim.x*blockDim.x)
    {
        p_corr = params.set.nodes3[i].corr;
#ifdef TRACK_TRAVERSALS
//        p_nodes_accessed = params.set.nodes0[i].nodes_accessed;
//        p_nodes_truncated = params.set.nodes0[i].nodes_truncated;
#endif
        for (j = 0; j < DIM; j ++)
        {
            p_coord[j] = params.set.nodes0[i].coord[j].items.max;
        }

        STACK_INIT();
        stack[1] = params.root_index;
		
        while (sp >= 1)
        {
            cur_node_index = STACK_TOP_NODE_INDEX;
            //cur_mask = STACK_TOP_MASK;
            CUR_NODE0 = params.tree.nodes0[cur_node_index];
			CUR_NODE1 = params.tree.nodes1[cur_node_index];

			if (CUR_NODE1.depth == SPLICE_DEPTH)
			{
				params.relation_matrix[CUR_NODE1.pre_id * params.npoints + i] = 1;
			}

#ifdef TRACK_TRAVERSALS
            p_nodes_accessed ++;
#endif
            STACK_POP();

            // inline call: can_correlate(...)
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

            if (sqrt(sum) - sqrt(boxsum) < rad)
            {
#ifdef TRACK_TRAVERSALS
//				p_nodes_needed ++; // this line is abandoned because we do not use this variable any more.
#endif

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
				}
				else
				{
                    CUR_NODE2 = params.tree.nodes2[cur_node_index];
					if (CUR_NODE1.depth < SPLICE_DEPTH)
					{
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

//		params.set.nodes3[i].corr = p_corr;
#ifdef TRACK_TRAVERSALS
//        params.set.nodes0[i].nodes_accessed = p_nodes_accessed;
//        params.set.nodes0[i].nodes_needed = p_nodes_needed;
#endif
    }
}







