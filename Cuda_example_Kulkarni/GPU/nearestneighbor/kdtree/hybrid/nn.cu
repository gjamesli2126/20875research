/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include "../../../common/util_common.h"
#include "nn_data_types.h"
#include "nn_functions.h"
#include "nn_pre_kernel.h"
#include "nn_kernel.h"
#include "nn_mem.h"
#include <assert.h>

const int K = 1;	// only support K = 1 for now, ignore command line value
int sort_flag = 0;
int verbose_flag = 0;
int check_flag = 0;
int ratio_flag = 0;
int warp_flag = 0;
int nthreads = 1;

int hTimes = 0;

Point *training_points;
KDCell *root;
Point *search_points;
int sort_split;
int npoints;
int nsearchpoints;

gpu_tree *h_tree;
gpu_tree *d_tree;
gpu_point *h_training_points;
gpu_point *d_training_points;
gpu_point *h_search_points;
gpu_point *d_search_points;
unsigned int COLS = 0;
unsigned int ROWS = 0;
int* h_matrix;
int* d_matrix;
dim3 blocks(NUM_OF_BLOCKS);
dim3 tpb(NUM_OF_THREADS_PER_BLOCK);
static int DEPTH_STAT = 1;
const int nStreams = 4;
vector<int>::iterator cluster_iter;

TIME_INIT(overall);
TIME_INIT(traversal);
TIME_INIT(init_kernel);
TIME_INIT(pre_calc);
TIME_INIT(kernel);
TIME_INIT(build_tree);
TIME_INIT(read_data);
TIME_INIT(CPU);
TIME_INIT(extra);

void check_depth(KDCell * node, int _depth);
void matrix_transform();
void Sort(int _count, vector<int> &_upper, int** _buffer, int _offset, int _level);

int main(int argc, char **argv) 
{
    TIME_START(overall);
	TIME_START(read_data);
	read_input(argc, argv);
	TIME_END(read_data);
	TIME_ELAPSED_PRINT(read_data, stdout);
	printf("configuration: K = %d DIM = %d npoints = %d nsearchpoints = %d\n", K, DIM, npoints, nsearchpoints);

	if(sort_flag)
    {
		sort_points(search_points, 0, nsearchpoints - 1, 0);
	}
	root = construct_tree(training_points, 0, npoints - 1, 0, 1);
	
	TIME_START(extra);
	check_depth(root, 0);
	TIME_END(extra);

	TIME_START(traversal);

	TIME_START(init_kernel);
	init_kernel<<<1, 1>>>();
	TIME_END(init_kernel);
	TIME_ELAPSED_PRINT(init_kernel, stdout);

	TIME_START(build_tree);

//	h_tree = build_gpu_tree(root);
	h_tree = gpu_transform_tree(root);
	h_training_points = gpu_transform_points(training_points, npoints);
	h_search_points = gpu_transform_points(search_points, nsearchpoints);

	d_tree = copy_tree_to_dev(h_tree);
	gpu_free_tree_host(h_tree);
	d_training_points = gpu_copy_points_to_dev(h_training_points, npoints);
	gpu_free_points_host(h_training_points);
	d_search_points = gpu_copy_points_to_dev(h_search_points, nsearchpoints);

	TIME_END(build_tree);
	TIME_ELAPSED_PRINT(build_tree, stdout);

	TIME_RESTART(extra);
    ROWS = nsearchpoints;
	COLS = DEPTH_STAT;
    int nMatrixSize = ROWS * COLS;
//    printf("ROWS = %d, COLS = %d, nMatrixSize = %d.\n", ROWS, COLS, nMatrixSize);

    SAFE_MALLOC(h_matrix, sizeof(int) * nMatrixSize);
    CUDA_SAFE_CALL(cudaMalloc(&(d_matrix), sizeof(int) * nMatrixSize));    
    kernel_params params;
    params.d_tree = *d_tree;
    params.d_training_points = d_training_points;
    params.n_training_points = nsearchpoints;
    params.d_search_points = d_search_points;
    params.n_search_points = nsearchpoints;
    params.d_array_points = NULL;
    params.n_root_index = 1;

	cudaStream_t stream[nStreams];
	cudaEvent_t startEvent, stopEvent;
	float ms; // elapsed time in milliseconds
	CUDA_SAFE_CALL( cudaEventCreate(&startEvent) );
	CUDA_SAFE_CALL( cudaEventCreate(&stopEvent) );
	for (int i = 0; i < nStreams; ++i) {
		CUDA_SAFE_CALL( cudaStreamCreate(&stream[i]) );
	}
	int stream_workload = nsearchpoints / nStreams;
	int point_offset = 0;
	int matrix_workload = nMatrixSize / nStreams;
	CUDA_SAFE_CALL( cudaEventRecord(startEvent,0) );
	for (int i = 0; i < nStreams; i ++) {
		pre_nearest_neighbor_search<<<blocks, tpb, 0, stream[i]>>>(params, d_matrix, point_offset, point_offset + stream_workload, DEPTH_STAT);
		point_offset += stream_workload;
	}
	for (int i = 0; i < nStreams; i ++) {
		CUDA_SAFE_CALL(cudaMemcpyAsync(&h_matrix[matrix_workload * i], &d_matrix[matrix_workload * i], matrix_workload * sizeof(int), cudaMemcpyDeviceToHost, stream[i]));
	}
	CUDA_SAFE_CALL( cudaEventRecord(stopEvent, 0) );
	CUDA_SAFE_CALL( cudaEventSynchronize(stopEvent) );
	CUDA_SAFE_CALL( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
	printf("Time for asynchronous transfer and execute (ms): %f\n", ms);

/*	for(int row = 0; row < npoints; row ++)
    {
		printf("%d:	", row);
		for (int col = 0; col < DEPTH_STAT; col ++)
			printf("%d ", h_matrix[col * npoints + row ]);
		printf("\n");
    }*/

	TIME_START(CPU);
	matrix_transform();
	TIME_END(CPU);
	TIME_ELAPSED_PRINT(CPU, stdout);

    gpu_copy_points_to_host(d_search_points, h_search_points, search_points, nsearchpoints);
	TIME_END(traversal);

	int correct_cnt = 0;
	for(int i = 0; i < nsearchpoints; i++) 
	{
		if(search_points[i].closest >= 0) 
		{
			if (training_points[search_points[i].closest].label == search_points[i].label) 
			{
				correct_cnt++;
			}
		}
	}

	float correct_rate = (float) correct_cnt / nsearchpoints;
	printf("correct rate: %.4f\n", correct_rate);

#ifdef TRACK_TRAVERSALS
	long sum_nodes_traversed = 0;
    int maximum = 0, all = 0;
    int num_of_warps = 0;
    unsigned long long maximum_sum = 0, all_sum = 0;
    for (int i = 0; i < nsearchpoints + (nsearchpoints % 32); i+=32) {
        int na = search_points[i].num_nodes_traversed;
//      printf("nodes warp %d: %d\n", i/32, na);
        sum_nodes_traversed += search_points[i].num_nodes_traversed;
        
        if (warp_flag) {
            maximum = na;
            all = na;
            for(int j = i + 1; j < i + 32 && j < nsearchpoints; j++) {
                sum_nodes_traversed += search_points[j].num_nodes_traversed;
                if(search_points[j].num_nodes_traversed)
                    na = search_points[j].num_nodes_traversed;

                if(search_points[j].num_nodes_traversed > maximum)
                    maximum = search_points[j].num_nodes_traversed;
                all += search_points[j].num_nodes_traversed;
            }

//            printf("%d\n", maximum);
            maximum_sum += maximum;
            all_sum += all;
            num_of_warps ++;
        }
    }
	
/*	for (int i = 0; i < nsearchpoints; i++)
    {
		Point *p = &search_points[i];
		sum_nodes_traversed += p->num_nodes_traversed;
	}*/
    printf("avg num of traversed nodes per warp is %f\n", (float)maximum_sum / num_of_warps);
    printf("@ sum_nodes_traversed: %ld\n", sum_nodes_traversed);
	printf("@ avg_nodes_traversed: %f\n", (float)sum_nodes_traversed / nsearchpoints);
#endif

	delete [] training_points;
	delete [] search_points;
	free_tree(root);
	gpu_free_points_host(h_search_points);
	gpu_free_points_dev(d_search_points);
	gpu_free_points_dev(d_training_points);
	free_tree_dev(d_tree);
	free(h_matrix);

	TIME_END(overall);
	TIME_ELAPSED_PRINT(traversal, stdout);
	TIME_ELAPSED_PRINT(extra, stdout);
	TIME_ELAPSED_PRINT(kernel, stdout);
	TIME_ELAPSED_PRINT(overall, stdout);

	return 0;
}

void check_depth(KDCell * node, int _depth)
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

void matrix_transform()
{
	int bytes = nsearchpoints * sizeof(int);
	int* cpu_buffer;
	int* gpu_buffer;
	SAFE_MALLOC(cpu_buffer, bytes);
	memset(cpu_buffer, 0, bytes);
    CUDA_SAFE_CALL(cudaMalloc(&(gpu_buffer), bytes));

	vector<int> DATA;
	DATA.reserve(nsearchpoints);
	for (int id = 0; id < nsearchpoints; id ++)
	{
		DATA.push_back(id);
	}
	Sort(nsearchpoints, DATA, &cpu_buffer, 0, 0);
	CUDA_SAFE_CALL(cudaMemcpy(gpu_buffer, cpu_buffer, bytes, cudaMemcpyHostToDevice));
	TIME_END(extra);

	kernel_params params;
	params.d_tree = *d_tree;
	params.d_training_points = d_training_points;
	params.n_training_points = nsearchpoints;
	params.d_search_points = d_search_points;
	params.n_search_points = nsearchpoints;
	params.d_array_points = gpu_buffer;
	params.n_root_index = 1;
	
	TIME_END(CPU);
	TIME_START(kernel);
	nearest_neighbor_search<<<blocks, tpb>>>(params);
	cudaError_t err = cudaThreadSynchronize();
	if(err != cudaSuccess) 
	{
		fprintf(stderr,"Kernel failed with error: %s\n", cudaGetErrorString(err));
		exit(1);
	}
	TIME_END(kernel);
	TIME_RESTART(CPU);
	
    free(cpu_buffer);
	CUDA_SAFE_CALL(cudaFree(gpu_buffer));
}

void Sort(int _count, vector<int> &_upper, int** _buffer, int _offset, int _level)
{
	vector<int>* clusters;
	clusters = new vector<int> [DEPTH_STAT];
	int temp = 0;
	int pos = 0;
	int buffer_index = _offset;
	for(int point = 0; point < _count; point ++)
	{
		pos = _upper[point] * DEPTH_STAT + _level;
		temp = h_matrix[pos];
		if (temp != -1)
		{
			clusters[temp].push_back(_upper[point]);
		}
		else
		{
			(*_buffer)[buffer_index ++] = *cluster_iter;
		}
	}
	
    int bytes = _count * sizeof(int);
	for(int group = 0; group < DEPTH_STAT; group ++)
	{
		if ( clusters[group].size() != 0)
		{
			//if (clusters[group].size() < 1000 || _level >= SPLICE_DEPTH)
			// 16 = sqrt(DEPTH_STAT)
			if (_level >= 16 || _level >= DEPTH_STAT - 1 || clusters[group].size() <= 32)
            {
				for(cluster_iter = clusters[group].begin(); cluster_iter != clusters[group].end(); cluster_iter ++)
				{
					(*_buffer)[buffer_index ++] = *cluster_iter;
				}
			}
			else
			{
                Sort(clusters[group].size(), clusters[group], _buffer, buffer_index, _level + 2);
                buffer_index += clusters[group].size();
			}
		}
	}
	for(int i = 0; i < DEPTH_STAT; i ++)
	{
		clusters[i].clear();
	}
	delete [] clusters;
}
