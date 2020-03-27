/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
//
//  main.cpp
//  BallTree-kNN

#include "knn_common.h"
#include "knn_functions.h"
#include "knn_kernel.h"
#include "knn_pre_kernel.h"

int sort_flag = 0;
int check_flag = 0;
int verbose_flag = 0;
int warp_flag = 0;
int ratio_flag = 0;
unsigned int npoints = 0;
unsigned int nsearchpoints = 0;

unsigned int K = 1;

datapoint *points = NULL;
datapoint *search_points = NULL;
float* nearest_distance = NULL;
int* nearest_point_index = NULL;

node* tree = NULL;
unsigned int max_depth = 0;
unsigned int nnodes = 0;

TIME_INIT(read_data);
TIME_INIT(build_tree);
TIME_INIT(sort);
TIME_INIT(extra);
TIME_INIT(traversal);
TIME_INIT(kernel);
TIME_INIT(pre_kernel);
TIME_INIT(CPU);
int* cpu_buffer;

void check_depth(node * root, int _depth);
void Sort(int _count, vector<int> &_upper, int** _buffer, int _offset, int _level);
static int DEPTH_STAT = 1;
const int nStreams = 4;
int *h_correlation_matrix = NULL;
int *d_correlation_matrix = NULL;
vector<int>::iterator cluster_iter;
vector<int>* DATA;

int main(int argc, char * argv[]) {
    
    TIME_START(read_data);
    read_input(argc, argv);
    printf("read done!\n");
    TIME_END(read_data);
    datapoint** dataList = new datapoint* [npoints];
    for (int i = 0; i < npoints; i ++) {
        dataList[i] = &points[i];
    }
    TIME_START(build_tree);
    tree = construct_tree(points, 0, npoints - 1, dataList, 0, 1);
    printf("construct done!\n");
    TIME_END(build_tree);
    printf("The max depth is %d, the nodes number is %d.\n", max_depth, nnodes);
//    printTree(tree, 0);
    
	TIME_START(extra);
	check_depth(tree, 0);
	DEPTH_STAT --;
	TIME_END(extra);

    TIME_START(sort);
    if (sort_flag) {
        sort_search_points(search_points, 0, nsearchpoints);
    }
    TIME_END(sort);
    
	init_kernel<<<1, 1>>>();
	gpu_tree * h_tree = gpu_transform_tree(tree);
	gpu_tree * d_tree = gpu_copy_to_dev(h_tree);

	// Allocate variables to store results of each thread
	datapoint * d_search_points;
	float * d_nearest_distance;
	int * d_nearest_point_index;
	// Read from but not written to
	CUDA_SAFE_CALL(cudaMalloc(&d_search_points, sizeof(datapoint)*nsearchpoints));
	CUDA_SAFE_CALL(cudaMemcpy(d_search_points, search_points, sizeof(datapoint)*nsearchpoints, cudaMemcpyHostToDevice));

	// Immediatly written to at kernel
	CUDA_SAFE_CALL(cudaMalloc(&d_nearest_distance, sizeof(float)*nsearchpoints*K));
	CUDA_SAFE_CALL(cudaMalloc(&d_nearest_point_index, sizeof(int)*nsearchpoints*K));

    TIME_START(traversal);
	//gpu_print_tree_host(h_tree);
	dim3 blocks(NUM_OF_BLOCKS);
	dim3 tpb(NUM_OF_THREADS_PER_BLOCK);
	unsigned int smem_bytes = DIM*NUM_OF_THREADS_PER_BLOCK*sizeof(float) + K*NUM_OF_THREADS_PER_BLOCK*sizeof(int) + K*NUM_OF_THREADS_PER_BLOCK*sizeof(float);

	// added by Cambridge
	TIME_RESTART(extra);
	long nMatrixSize = nsearchpoints * DEPTH_STAT;
	printf("nsearchpoints = %d, DEPTH_STAT = %d, nMatrixSize = %d.\n", nsearchpoints, DEPTH_STAT, nMatrixSize);
	SAFE_MALLOC(h_correlation_matrix, sizeof(int) * nMatrixSize);  
    CUDA_SAFE_CALL(cudaMalloc(&(d_correlation_matrix), sizeof(int) * nMatrixSize));

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
		k_nearest_neighbor_pre_search<<<blocks, tpb, 0, stream[i]>>>(*d_tree, nsearchpoints, d_search_points, d_nearest_distance, d_nearest_point_index, K, d_correlation_matrix, point_offset, point_offset + stream_workload, DEPTH_STAT);
		point_offset += stream_workload;
	}
	for (int i = 0; i < nStreams; i ++) {
		CUDA_SAFE_CALL(cudaMemcpyAsync(&h_correlation_matrix[matrix_workload * i], &d_correlation_matrix[matrix_workload * i], matrix_workload * sizeof(int), cudaMemcpyDeviceToHost, stream[i]));
	}
	CUDA_SAFE_CALL( cudaEventRecord(stopEvent, 0) );
	CUDA_SAFE_CALL( cudaEventSynchronize(stopEvent) );
	CUDA_SAFE_CALL( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
	printf("Time for asynchronous transfer and execute (ms): %f\n", ms);
	
    int bytes = nsearchpoints * sizeof(int);
    int buffer_index = 0;
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

    TIME_START(kernel);
	k_nearest_neighbor_search<<<blocks, tpb, 0>>>(*d_tree, nsearchpoints, d_search_points, d_nearest_distance,
			d_nearest_point_index, K, gpu_buffer);

	cudaError_t err = cudaThreadSynchronize();
	if(err != cudaSuccess) {
		fprintf(stderr,"Kernel failed with error: %s\n", cudaGetErrorString(err));
		exit(1);
	}
    TIME_END(kernel);
//    for(int i = 0; i < nsearchpoints; i ++) {
//        k_nearest_neighbor_search(tree, &search_points[i], i*K);
//    }
	// Copy results back
	CUDA_SAFE_CALL(cudaMemcpy(nearest_point_index, d_nearest_point_index, sizeof(int)*nsearchpoints*K, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(nearest_distance, d_nearest_distance, sizeof(int)*nsearchpoints*K, cudaMemcpyDeviceToHost));

#ifdef TRACK_TRAVERSALS
	CUDA_SAFE_CALL(cudaMemcpy(search_points, d_search_points, sizeof(datapoint)*nsearchpoints, cudaMemcpyDeviceToHost));
#endif
	CUDA_SAFE_CALL(cudaFree(d_nearest_point_index));
	CUDA_SAFE_CALL(cudaFree(d_nearest_distance));
	CUDA_SAFE_CALL(cudaFree(d_search_points));

    TIME_END(traversal);
    print_result();
        
#ifdef TRACK_TRAVERSALS
	long long sum_nodes_traversed = 0;
    int maximum = 0, all = 0;
    int num_of_warps = 0;
    unsigned long long maximum_sum = 0, all_sum = 0;
	for (int i = 0; i < nsearchpoints + (nsearchpoints % 32); i+=32) {
		int na =search_points[i].numNodesTraversed;
//        printf("nodes warp %d: %d\n", i/32, na);
        sum_nodes_traversed += search_points[i].numNodesTraversed;

        if (warp_flag) {
            maximum = na;
            all = na;
            for(int j = i + 1; j < i + 32 && j < nsearchpoints; j++) {
		    	sum_nodes_traversed += search_points[j].numNodesTraversed;
	    		if(search_points[j].numNodesTraversed)
    				na = search_points[j].numNodesTraversed;
		
                    if(search_points[j].numNodesTraversed > maximum)
                        maximum = search_points[j].numNodesTraversed;
                    all += search_points[j].numNodesTraversed;       
            }

            printf("%d\n", maximum);
            maximum_sum += maximum;
            all_sum += all;
            num_of_warps ++;
        }
    }	
    printf("avg num of traversed nodes per warp is %f\n", (float)maximum_sum / num_of_warps);
	printf("avg nodes: %f\n", (float)sum_nodes_traversed / nsearchpoints);
#endif

    TIME_ELAPSED_PRINT(read_data, stdout);
    TIME_ELAPSED_PRINT(build_tree, stdout);
    TIME_ELAPSED_PRINT(extra, stdout);
    TIME_ELAPSED_PRINT(sort, stdout);
    TIME_ELAPSED_PRINT(kernel, stdout);
    TIME_ELAPSED_PRINT(traversal, stdout);
    
    delete [] points;
    delete [] search_points;
    delete [] dataList;
    
    return 0;
}

void check_depth(node * root, int _depth)
{
	if (root == NULL)
	{
		return;
	}

	root->depth = _depth;
	if (root->depth == SPLICE_DEPTH)
	{
		root->pre_id = DEPTH_STAT ++;
		return;
	}

	check_depth(root->left, _depth + 1);
	check_depth(root->right, _depth + 1);
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
		temp = h_correlation_matrix[pos];
		if (temp != -1)
		{
			clusters[temp].push_back(_upper[point]);
		}
		else
		{
			(*_buffer)[buffer_index ++] = _upper[point];
		}
	}


	int bytes = _count * sizeof(int);
	for(int group = 0; group < DEPTH_STAT; group ++)
	{
		if ( clusters[group].size() != 0)
		{
			//if (clusters[group].size() < 1000 || _level >= SPLICE_DEPTH)
			if (_level >= SPLICE_DEPTH || clusters[group].size() <= 32)
            {
				for(cluster_iter = clusters[group].begin(); cluster_iter != clusters[group].end(); cluster_iter ++)
				{
					(*_buffer)[buffer_index ++] = *cluster_iter;
				}
//			printf("level = %d, node = %d, size = %d, buffer_index = %d.\n", _level, group, clusters[group].size(), buffer_index);
			}
			else
			{
                Sort(clusters[group].size(), clusters[group], _buffer, buffer_index, _level + 1);
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
