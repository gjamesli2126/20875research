/* -*- mode: c -*- */
/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <unistd.h>
#include <time.h>

#include "../../../common/util_common.h"
#include "knn.h"
#include "knn_gpu.h"

node *points;
node *search_points;
//float *search_points;
unsigned int npoints;
unsigned int nsearchpoints;

float* nearest_distance;
unsigned int* nearest_point_index;

float* nearest_distance_brute;
unsigned int* nearest_point_index_brute;

node * tree;

int K;

int sort_flag = 0;
int verbose_flag = 0;
int check_flag = 0;
int ratio_flag = 0; // reserved for later use
int warp_flag = 0; // track the maximum length of each warp workload

TIME_INIT(construct_tree);
TIME_INIT(gpu_build_tree);
TIME_INIT(init_kernel);
TIME_INIT(gpu_copy_tree_to);
TIME_INIT(kernel);
TIME_INIT(gpu_copy_tree_from);
TIME_INIT(traversal_time);
TIME_INIT(runtime);
TIME_INIT(extra);

void check_depth(node * node, int _depth);
void track(node * node, int _depth);
static int DEPTH_STAT = 1;
const int nStreams = 4;
int *h_correlation_matrix = NULL;
int *d_correlation_matrix = NULL;
vector<int>::iterator cluster_iter;
void Sort(int _count, vector<int> &_upper, int* _buffer, int _offset, int _level);

int main(int argc, char **argv) {

	int i, j, k;
	float min = FLT_MAX;
	float max = FLT_MIN;
	int c;
	char *input_file;

	if(argc < 2) {
		fprintf(stderr, "usage: nn [-c] [-v] [-s] <k> <input_file> <npoints> [<nsearchpoints>]\n");
		exit(1);
	}

	while((c = getopt(argc, argv, "cvt:srw")) != -1) {
		switch(c) {
		case 'c':
			check_flag = 1;
			break;

		case 'v':
			verbose_flag = 1;
			break;

		case 's':
			sort_flag = 1;
			break;
        
        case 'r':
            ratio_flag = 1;
            break;
       
        case 'w':
            warp_flag = 1;
            break;

		case '?':
			fprintf(stderr, "Error: unknown option.\n");
			exit(1);
			break;

		default:
			abort();
		}
	}
	
	for(i = optind; i < argc; i++) {
		switch(i - optind) {
		case 0:
			K = atoi(argv[i]);
			if(K <= 0) {
				fprintf(stderr, "Invalid number of neighbors.\n");
				exit(1);
			}
			break;

		case 1:
			input_file = argv[i];
			break;

		case 2:
				npoints = atoi(argv[i]);
				nsearchpoints = npoints;
				if(npoints <= 0) {
					fprintf(stderr, "Not enough points.\n");
					exit(1);
				}
				break;

		case 3:
			nsearchpoints = atoi(argv[i]);
			if(nsearchpoints <= 0) {
				fprintf(stderr, "Not enough search points.");
				exit(1);
			}
			break;
		}
	}
	
	printf("configuration: sort_flag=%d, check_flag = %d, verbose_flag=%d, K=%d, input_file=%s, npoints=%d, nsearchpoints=%d\n", sort_flag, check_flag, verbose_flag, K, input_file, npoints, nsearchpoints);
	
	SAFE_CALLOC(points, npoints, sizeof(node));
	//SAFE_MALLOC(search_points, sizeof(float)*nsearchpoints*DIM);
	SAFE_CALLOC(search_points, nsearchpoints, sizeof(node));
	SAFE_MALLOC(nearest_distance, sizeof(float)*nsearchpoints*K);
	SAFE_MALLOC(nearest_point_index, sizeof(unsigned int)*nsearchpoints*K);

	if(check_flag) {
		SAFE_MALLOC(nearest_distance_brute, sizeof(float)*nsearchpoints*K);
		SAFE_MALLOC(nearest_point_index_brute, sizeof(unsigned int)*nsearchpoints*K);
	}

	if(strcmp(input_file, "random") != 0) {
		FILE * in = fopen(input_file, "r");
		if(in == NULL) {
			fprintf(stderr, "Could not open %s\n", input_file);
			exit(1);
		}
		
		int junk;
		float data;

		for(i = 0; i < npoints; i++) {
			points[i].point_index = i;
			if(fscanf(in, "%d", &junk) != 1) {
				fprintf(stderr, "Input file not large enough.\n");
				exit(1);
			} 
			for(j = 0; j < DIM; j++) {
				if(fscanf(in, "%f", &data) != 1) {
					fprintf(stderr, "Input file not large enough.\n");
					exit(1);
				}
				points[i].point[j] = data;
			}
		}
	
		for(i = 0; i < nsearchpoints; i++) {
			if(fscanf(in, "%d", &junk) != 1) {
				fprintf(stderr, "Input file not large enough.\n");
				exit(1);
			}
			for(j = 0; j < DIM; j++) {
				if(fscanf(in, "%f", &data) != 1) {
					fprintf(stderr, "Input file not large enough.\n");
					exit(1);
				}
				search_points[i].point[j] = data;
				//search_points[i*DIM + j] = data;
			}
		}
		
		fclose(in);
		
	} else {
		for(i = 0; i < npoints; i++) {
			points[i].point_index = i;			
			for(j = 0; j < DIM; j++) {
				points[i].point[j] = 1.0 + (float)rand() / RAND_MAX;			
			}
		}
	
		for(i = 0; i < nsearchpoints; i++) {
			for(j = 0; j < DIM; j++) {
				//search_points[i*DIM + j] = 1.0 + (float)rand() / RAND_MAX;			
				search_points[i].point[j] = 1.0 + (float)rand() / RAND_MAX;
			}
		}
	}

	TIME_START(runtime);
	TIME_START(construct_tree);

	if(sort_flag) {
		construct_tree(search_points, 0, nsearchpoints - 1, 0);
    }
	
	tree = construct_tree(points, 0, npoints - 1, 0);

	TIME_END(construct_tree);
	
	TIME_START(extra);
	check_depth(tree, 0);
	DEPTH_STAT--;
	TIME_END(extra);

	TIME_START(traversal_time);

	TIME_START(gpu_build_tree);
	// *** GPU Kerel Call *** //
	gpu_tree * h_tree = gpu_transform_tree(tree);
	TIME_END(gpu_build_tree);
	
	TIME_START(init_kernel);
	init_kernel<<<1, 1>>>();
	TIME_END(init_kernel);

	TIME_START(gpu_copy_tree_to);
	gpu_tree * d_tree = gpu_copy_to_dev(h_tree);

	// Allocate variables to store results of each thread
	node * d_search_points;
	float * d_nearest_distance;
	int * d_nearest_point_index;

	#ifdef TRACK_TRAVERSALS	
	int *h_nodes_accessed;
	int *d_nodes_accessed;
	SAFE_CALLOC(h_nodes_accessed, nsearchpoints, sizeof(int));
	CUDA_SAFE_CALL(cudaMalloc(&d_nodes_accessed, sizeof(int)*nsearchpoints));
	CUDA_SAFE_CALL(cudaMemcpy(d_nodes_accessed, h_nodes_accessed, sizeof(int)*nsearchpoints, cudaMemcpyHostToDevice));
	#endif

	// Read from but not written to
	CUDA_SAFE_CALL(cudaMalloc(&d_search_points, sizeof(node)*nsearchpoints));
	CUDA_SAFE_CALL(cudaMemcpy(d_search_points, search_points, sizeof(node)*nsearchpoints, cudaMemcpyHostToDevice));

	// Immediatly written to at kernel
	CUDA_SAFE_CALL(cudaMalloc(&d_nearest_distance, sizeof(float)*nsearchpoints*K));
	CUDA_SAFE_CALL(cudaMalloc(&d_nearest_point_index, sizeof(int)*nsearchpoints*K));
	
	TIME_END(gpu_copy_tree_to);

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
		nearest_neighbor_pre_search<<<blocks, tpb, 0, stream[i]>>>(*d_tree, nsearchpoints, d_search_points, d_nearest_distance, d_nearest_point_index, K, d_correlation_matrix, point_offset, point_offset + stream_workload, DEPTH_STAT
#ifdef TRACK_TRAVERSALS
	, d_nodes_accessed
#endif
	);
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
	Sort(nsearchpoints, DATA, cpu_buffer, 0, 0);

	CUDA_SAFE_CALL(cudaMemcpy(gpu_buffer, cpu_buffer, bytes, cudaMemcpyHostToDevice));

	
	TIME_END(extra);
	// Kernel
	TIME_START(kernel);
	nearest_neighbor_search<<<blocks, tpb, smem_bytes>>>(*d_tree, nsearchpoints, d_search_points, d_nearest_distance, d_nearest_point_index, K, gpu_buffer
#ifdef TRACK_TRAVERSALS
	, d_nodes_accessed	
#endif
	);
																											 
	cudaError_t err = cudaThreadSynchronize();
	if(err != cudaSuccess) {
		fprintf(stderr,"Kernel failed with error: %s\n", cudaGetErrorString(err));
		exit(1);
	}
	
	TIME_END(kernel);
	TIME_START(gpu_copy_tree_from);

	// Copy results back
	CUDA_SAFE_CALL(cudaMemcpy(nearest_point_index, d_nearest_point_index, sizeof(int)*nsearchpoints*K, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(nearest_distance, d_nearest_distance, sizeof(int)*nsearchpoints*K, cudaMemcpyDeviceToHost));

	#ifdef TRACK_TRAVERSALS
	CUDA_SAFE_CALL(cudaMemcpy(h_nodes_accessed, d_nodes_accessed, sizeof(int)*nsearchpoints, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_nodes_accessed));
	#endif
	CUDA_SAFE_CALL(cudaFree(d_nearest_point_index));
	CUDA_SAFE_CALL(cudaFree(d_nearest_distance));
	CUDA_SAFE_CALL(cudaFree(d_search_points));

	TIME_END(gpu_copy_tree_from);
	TIME_END(traversal_time);
	TIME_END(runtime);

	#ifdef TRACK_TRAVERSALS
		unsigned long long sum_nodes_accessed = 0;
/*	  int nwarps = 0;
	for(i = 0; i < nsearchpoints + (nsearchpoints % 32); i+=32, nwarps++) {			
		sum_nodes_accessed += (unsigned long)h_nodes_accessed[i];
//		printf("nodes warp %d: %d\n", i/32, h_nodes_accessed[i]);
	}
	printf("avg nodes: %f\n", (float)sum_nodes_accessed/nwarps);*/

    if (warp_flag) {
	    int maximum = 0, all = 0;
        int num_of_warps = 0;
        unsigned long long maximum_sum = 0, all_sum = 0;
    	for(i = 0; i < nsearchpoints + (nsearchpoints % 32); i+=32) {
	    	int na = h_nodes_accessed[i];
            maximum = na;
            all = na;

	    	for(j = i + 1; j < i + 32 && j < nsearchpoints; j++) {
                if(h_nodes_accessed[j] > maximum)
                    maximum = h_nodes_accessed[j];
                all += h_nodes_accessed[j];
	        }
            printf("%d\n", maximum);
            maximum_sum += maximum;
            all_sum += all;
            num_of_warps ++;
    	}
        printf("avg num of traversed nodes per warp is %f\n", (float)maximum_sum / num_of_warps);
    }

	unsigned long long sum_nodes_traversed = 0;
	for (int i = 0; i < nsearchpoints; i++)
    {
		sum_nodes_traversed += h_nodes_accessed[i];
	}
    printf("@ sum_nodes_traversed: %ld\n", sum_nodes_traversed);
	printf("@ avg_nodes_traversed: %f\n", (float)sum_nodes_traversed / nsearchpoints);	
	#endif

	if(verbose_flag) {
		for(j = 0; j < nsearchpoints; j++) {
//			printf("\n%d: %f %f %f --- ", j, search_points[j].point[0], search_points[j].point[1], search_points[j].point[2]);
            printf("\n%d: ", j);
            for(i = 0; i < K; i++) {
				if(i == K-1)
					printf("%d (%1.3f)", nearest_point_index[j*K + i], nearest_distance[j*K + i]);
				else
					printf("%d (%1.3f),", nearest_point_index[j*K + i], nearest_distance[j*K + i]);
			}
		}
		printf("\n");
	}
	
	TIME_ELAPSED_PRINT(construct_tree, stdout);
	TIME_ELAPSED_PRINT(gpu_build_tree, stdout);
	TIME_ELAPSED_PRINT(init_kernel, stdout);
	TIME_ELAPSED_PRINT(gpu_copy_tree_to, stdout);
	TIME_ELAPSED_PRINT(extra, stdout);
	TIME_ELAPSED_PRINT(kernel, stdout);
    TIME_ELAPSED_PRINT(gpu_copy_tree_from, stdout);
	TIME_ELAPSED_PRINT(traversal_time, stdout);
	TIME_ELAPSED_PRINT(runtime, stdout);

	gpu_free_tree_dev(d_tree);
	gpu_free_tree_host(h_tree);
	
	return 0;
}

void check_depth(node * node, int _depth)
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

void Sort(int _count, vector<int> &_upper, int* _buffer, int _offset, int _level)
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
			_buffer[buffer_index ++] = _upper[point];
        }
	}

	int bytes = _count * sizeof(int);
	for(int group = 0; group < DEPTH_STAT; group ++)
	{
		if ( clusters[group].size() != 0)
		{
			if (_level >= 16 || _level >= DEPTH_STAT - 1 || clusters[group].size() <= 32)
            {
				for(cluster_iter = clusters[group].begin(); cluster_iter != clusters[group].end(); cluster_iter ++)
				{
					_buffer[buffer_index ++] = *cluster_iter;
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


