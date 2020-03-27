/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
//#define OVERLAPPING

#include "pc.h"

int npoints = 0; // number of input points
kd_cell **points = NULL; // input points
kd_cell *root = NULL; // root of the tree

int sort_flag = 0;
int verbose_flag = 0;
int check_flag = 0;
int ratio_flag = 0;
int warp_flag = 0;
int nthreads = 0;

unsigned long corr_sum = 0;
unsigned int sum_of_nodes = 0;

#ifdef TRACK_TRAVERSALS
unsigned long long sum_nodes_traversed = 0;
unsigned long long nodes_needed_sum = 0;
#endif

gpu_tree *h_root = NULL; // root of the host GPU tree
gpu_tree d_root;

gpu_tree *h_pre_root = NULL; // root of the pre GPU tree
gpu_tree d_pre_root;

bool *h_correlation_matrix;
bool *d_correlation_matrix;
unsigned int COLS = 0;
unsigned int ROWS = 0;

int sort_split = 0; // axis component compared
int sortidx = 0;

gpu_point_set *h_set = NULL; // root of the host GPU point set
gpu_point_set d_set;

cudaError_t e;
dim3 tpb(NUM_OF_THREADS_PER_BLOCK);
dim3 nb(NUM_OF_BLOCKS);

TIME_INIT(overall);
TIME_INIT(read_input);
TIME_INIT(build_tree);
TIME_INIT(init_kernel);
TIME_INIT(find_correlation);
TIME_INIT(kernel);

TIME_INIT(extra);
void check_depth(kd_cell * node, int _depth);
static int DEPTH_STAT = 0;
const int nStreams = 4;

int main(int argc, char* argv[]) 
{
#ifdef TRACK_TRAVERSALS
//    cudaDeviceProp prop;
//    cudaGetDeviceProperties(&prop, 0);
//    printf("Device : %s. The splice depth is %d.\n", prop.name, SPLICE_DEPTH);
#endif
	int i = 0; // loop variable
	int j = 0; // loop variable

	srand(0); // for quicksort

	TIME_START(overall);

    TIME_START(init_kernel);
    init_kernel<<<1, 1>>>();
    TIME_END(init_kernel);

	TIME_START(read_input);
	read_input(argc, argv);
	TIME_END(read_input);
	
	TIME_START(build_tree);
	// cpu tree
	root = build_tree(points, 0, 0, npoints - 1, 1);

	TIME_START(extra);
	check_depth(root, 0);
	TIME_END(extra);

	// gpu tree
    h_root = build_gpu_tree(root); // build up the gpu tree
    d_root.nnodes = h_root->nnodes;
	d_root.tree_depth = h_root->tree_depth;
    alloc_tree_dev(h_root, &d_root);
    copy_tree_to_dev(h_root, &d_root);
	TIME_END(build_tree);

	if (!sort_flag)
	{
		srand(0);
		for (i = 0; i < npoints; i ++)
		{
			j = rand() % npoints;
			kd_cell *temp = points[i];
			points[i] = points[j];
			points[j] = temp;
		}
	}

    SAFE_MALLOC(h_set, sizeof(gpu_point_set));
    h_set->npoints = npoints;
    SAFE_MALLOC(h_set->nodes0, sizeof(gpu_node0) * npoints);
    SAFE_MALLOC(h_set->nodes3, sizeof(gpu_node3) * npoints);
    for (i = 0 ; i < h_set->npoints; i++)
    {
#ifdef TRACK_TRAVERSALS
        h_set->nodes0[i].nodes_accessed = points[i]->nodes_accessed;
        h_set->nodes0[i].nodes_truncated = points[i]->nodes_truncated;
#endif
        for (j = 0; j < DIM; j ++)
        {
            h_set->nodes0[i].coord[j].items.max = points[i]->coord_max[j];
            h_set->nodes0[i].coord[j].items.min = points[i]->min[j];
        }

        h_set->nodes3[i].corr = points[i]->corr;
//        h_set->nodes3[i].cpu_addr = NULL;
		h_set->nodes3[i].point_id = points[i]->id;
    }
    d_set.npoints = h_set->npoints;
//	CUDA_SAFE_CALL(cudaMalloc(&(d_set.nodes0), sizeof(gpu_node0)*npoints));
//	CUDA_SAFE_CALL(cudaMalloc(&(d_set.nodes3), sizeof(gpu_node3)*npoints));
//    alloc_set_dev(h_set, &d_set);

// Pre compute correlation matrix

	TIME_RESTART(extra);
    ROWS = npoints;
    COLS = DEPTH_STAT;
    long nMatrixSize = ROWS * COLS;
	SAFE_MALLOC(h_correlation_matrix, sizeof(bool) * nMatrixSize);
    CUDA_SAFE_CALL(cudaMalloc(&(d_correlation_matrix), sizeof(bool) * nMatrixSize));
	TIME_END(extra);

	alloc_set_dev(h_set, &d_set);
    copy_set_to_dev(h_set, &d_set);

	TIME_RESTART(extra);
	pc_pre_kernel_params d_params;
    d_params.tree = d_root;
	d_params.root_index = 1;
    d_params.set = d_set;
	d_params.relation_matrix = d_correlation_matrix;
    d_params.rad = RADIUS;
    d_params.npoints = d_set.npoints;
	printf("ROWS = %d, COLS = %d.\n", ROWS, COLS);

/*    // ** Pre Kernel ** //
	pre_compute_correlation<<<nb, tpb>>>(d_params);
    cudaThreadSynchronize();
    e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        fprintf(stderr, "error: kernel error: %s.\n", cudaGetErrorString(e));
        exit(1);
    }

	CUDA_SAFE_CALL(cudaMemcpy(h_correlation_matrix, d_correlation_matrix, sizeof(bool) * nMatrixSize, cudaMemcpyDeviceToHost));*/


	cudaStream_t stream[nStreams];
	cudaEvent_t startEvent, stopEvent;
	float ms; // elapsed time in milliseconds
	CUDA_SAFE_CALL( cudaEventCreate(&startEvent) );
	CUDA_SAFE_CALL( cudaEventCreate(&stopEvent) );
	for (int i = 0; i < nStreams; ++i) {
		CUDA_SAFE_CALL( cudaStreamCreate(&stream[i]) );
	}
	long stream_workload = npoints / nStreams;
	long point_offset = 0;
	long matrix_workload = nMatrixSize / nStreams;
	CUDA_SAFE_CALL( cudaEventRecord(startEvent,0) );
	for (int i = 0; i < nStreams; i ++) {
		pre_compute_correlation<<<nb, tpb, 0, stream[i]>>>(d_params, point_offset, point_offset + stream_workload);
		point_offset += stream_workload;
	}
	for (long i = 0; i < nStreams; i ++) {
		CUDA_SAFE_CALL(cudaMemcpyAsync(&h_correlation_matrix[matrix_workload * i], &d_correlation_matrix[matrix_workload * i], matrix_workload * sizeof(bool), cudaMemcpyDeviceToHost, stream[i]));
	}
	CUDA_SAFE_CALL( cudaEventRecord(stopEvent, 0) );
	CUDA_SAFE_CALL( cudaEventSynchronize(stopEvent) );
	CUDA_SAFE_CALL( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
	printf("Time for asynchronous transfer and execute (ms): %f\n", ms);





    CUDA_SAFE_CALL(cudaFree(d_correlation_matrix));
    d_correlation_matrix = NULL;
// we can't delete the point set now, maybe can do more work later
//  free_set_dev(&d_set);

    TIME_START(find_correlation);
    find_correlation(0, npoints);
    TIME_END(find_correlation);
     
    copy_set_to_host(h_set, &d_set);
    for (i = 0; i < h_set->npoints; i ++)
    {
    #ifdef TRACK_TRAVERSALS
        points[i]->nodes_accessed = h_set->nodes0[i].nodes_accessed;
        points[i]->nodes_truncated = h_set->nodes0[i].nodes_truncated;
    #endif
        points[i]->corr = h_set->nodes3[i].corr;
//		points[i]->id = h_set->nodes3[i].point_id;
    }

    free(h_set->nodes0);
    free(h_set->nodes3);
    free(h_set);
    h_set = NULL;

    for(i = 0; i < npoints; i++)
	{
		corr_sum += (unsigned long)points[i]->corr;
#ifdef TRACK_TRAVERSALS
		sum_nodes_traversed += (unsigned long)points[i]->nodes_accessed;
//		nodes_truncated_sum += (unsigned long)points[i]->nodes_truncated;
#endif
	}
    printf("@ avg_corr: %f\n", (float)corr_sum / npoints);
#ifdef TRACK_TRAVERSALS
//	printf("@ sum_accessed: %ld\n", nodes_accessed_sum);
//	printf("@ sum_truncated: %ld\n", nodes_truncated_sum);
    
    if (warp_flag) {
        int maximum = 0, all = 0;
        unsigned long long maximum_sum = 0, all_sum = 0;
        int num_of_warps = 0;
        for(i = 0; i < npoints + (npoints % 32); i+=32) {
            int na = points[i]->nodes_accessed;
            maximum = na;
            all = na;

            for(j = i + 1; j < i + 32 && j < npoints; j++) {
                if(points[i]->nodes_accessed > maximum)
                    maximum = points[i]->nodes_accessed;
                all += points[i]->nodes_accessed;
            }
            printf("%d\n", maximum);
            maximum_sum += maximum;
            all_sum += all;
            num_of_warps ++;
        }
        printf("avg num of traversed nodes per warp is %f\n", (float)maximum_sum / num_of_warps);
    }
     
    printf("@ sum_nodes_traversed: %ld\n", sum_nodes_traversed);
	printf("@ avg_nodes_traversed: %f\n", (float)sum_nodes_traversed / npoints);
#endif
	
    free_tree_dev(&d_root);
    free_gpu_tree(h_root);
    h_root = NULL;
//	free(h_correlation_matrix);

    TIME_END(overall);
 
    TIME_ELAPSED_PRINT(overall, stdout);
    TIME_ELAPSED_PRINT(read_input, stdout);
    TIME_ELAPSED_PRINT(build_tree, stdout);
    TIME_ELAPSED_PRINT(init_kernel, stdout);    
    TIME_ELAPSED_PRINT(find_correlation, stdout);
    TIME_ELAPSED_PRINT(kernel, stdout);
	TIME_ELAPSED_PRINT(extra, stdout);
}

void check_depth(kd_cell * node, int _depth)
{
	if (!node)
	{
		return;
	}

	node->depth = _depth;
	if (node->depth == SPLICE_DEPTH)
	{
		node->pre_id = DEPTH_STAT ++;
		return;
	}

	check_depth(node->left, _depth + 1);
	check_depth(node->right, _depth + 1);
}

void find_correlation(int start, int end)
{
	int bytes = npoints * sizeof(int);
	int* gpu_buffer;
	CUDA_SAFE_CALL(cudaMalloc(&(gpu_buffer), bytes));

	int nnodes = COLS;
	
	pc_kernel_params d_params;
    d_params.tree = d_root;
    d_params.root_index = 1;			// changes every loop
    d_params.set = d_set;
	d_params.index_buffer = gpu_buffer;
    d_params.rad = RADIUS;
    d_params.npoints = npoints;		// changes every loop

	int buffer_index = 0;
	int* DATA = new int [npoints];
	int data_size = npoints;
	for(int i = 0; i < npoints; i ++) {
		DATA[i] = i;
	}

	for(int node = 0; node < nnodes; node += 2)
	{
//		buffer_index = 0;
//		memset(cpu_buffer, 0, bytes);

		int* A = new int [npoints];
		int* B = new int [npoints];
		int a_size = 0;
		int b_size = 0;
		long index = node * npoints;
		for (int i = 0; i < npoints; i ++) {
			if (h_correlation_matrix[index + DATA[i]]) {
				A[a_size ++] = DATA[i];
			} else {
				B[b_size ++] = DATA[i];
			}
		}
		for (int i = 0; i < b_size; i ++) {
			A[a_size ++] = B[i];
		}

		delete [] DATA;
		delete [] B;
		DATA = A;
	}

	CUDA_SAFE_CALL(cudaMemcpy(gpu_buffer, DATA, bytes, cudaMemcpyHostToDevice));
	delete [] DATA;
	DATA = NULL;

	d_params.root_index = 1;
	d_params.npoints = npoints;

//	printf("step: %d, npoint: %d.\n", d_params.root_index, d_params.npoints);
		
	TIME_END(extra);
	TIME_START(kernel);
	compute_correlation<<<nb, tpb>>>(d_params);
	cudaThreadSynchronize();
	e = cudaGetLastError();
	if (e != cudaSuccess)
	{
		fprintf(stderr, "error: kernel error: %s.\n", cudaGetErrorString(e));
		exit(1);
	}

	TIME_END(kernel);

}

