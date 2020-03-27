/* -*- mode: c++ -*- */
// A VP-Tree implementation, by Steve Hanov. (steve.hanov@gmail.com)
// Released to the Public Domain
// Based on "Data Structures and Algorithms for Nearest Neighbor Search" by Peter N. Yianilos
#include "ptrtab.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <float.h>
#include <getopt.h>
#include "../../../common/util_common.h"
#include <vector>
using namespace std;

#include "pc.h"
#include "ptrtab.h"

struct Point *__GPU_point_Point_d;
struct Point *__GPU_point_Point_h;
struct __GPU_point *__GPU_point_array_d;
struct __GPU_point *__GPU_point_array_h;
struct Point *__GPU_node_point_d;
struct Point *__GPU_node_point_h;
struct __GPU_tree __the_tree_h;
struct __GPU_tree __the_tree_d;

struct Point **_items;
int npoints;
int nsearchpoints;
int verbose_flag;
int check_flag;
int sort_flag;
int ratio_flag = 0;
int warp_flag = 0;
int nthreads;
struct Node *_root;

float radius = 0.03;
unsigned long corr_sum = 0;

TIME_INIT(search_kernel);
TIME_INIT(kernel);
TIME_INIT(pre_kernel);
TIME_INIT(init_kernel);
TIME_INIT(search);
TIME_INIT(extra);
TIME_INIT(CPU);
int* cpu_buffer;

void check_depth(Node * node, int _depth);
void Sort(int _count, vector<int> &_upper, int** _buffer, int _offset, int _level);
static int DEPTH_STAT = 1;
const int nStreams = 6;
int *h_correlation_matrix = NULL;
int *d_correlation_matrix = NULL;
vector<int>::iterator cluster_iter;
vector<int>* DATA;
struct Point** points;
struct Point** searchpoints;

int compare_point(const void *p1, const void *p2) {
  const struct Point *pp1 =  *((const struct Point **)p1);
  const struct Point *pp2 =  *((const struct Point **)p2);
  if ((pp1 -> vantage_dist) < (pp2 -> vantage_dist)) 
    return -1;
  else 
    return 1;
}

struct Node* buildFromPoints(struct Point **_items, int lower, int upper ) {
	int i, j;
	struct Point *ptmp;
	
	#ifdef TRACK_TRAVERSALS
	static int node_id = 0;
	#endif

	if ( upper == lower ) {
		return NULL;
	}

	struct Node* node = NULL;
	SAFE_MALLOC(node, sizeof(struct Node));

	node->point = _items[lower];
	node->left = NULL;
	node->right = NULL;
	node->parent = NULL;
	node->threshold = 0.0;
	#ifdef TRACK_TRAVERSALS
	node->id = node_id++;
	#endif

	if ( upper - lower > 1 ) {

		// choose an arbitrary point and move it to the start
		// This is one of several ways to find the best candidate VP
		i = lower; //(int)((float)rand() / RAND_MAX * (upper - lower - 1) ) + lower;
		ptmp = _items[lower];
		_items[lower] = _items[i];
		_items[i] = ptmp;

		int median = ( upper + lower ) / 2;

		// partitian around the median distance		
		for(i = lower + 1; i < upper; i++) {
			_items[i]->vantage_dist = mydistance(_items[lower], _items[i]);
		}

		qsort(&_items[lower + 1], upper - lower - 1, sizeof(struct Point *), compare_point);

		// what was the median?
		node->threshold = mydistance( _items[lower], _items[median]);

		node->point = _items[lower];
		node->left = buildFromPoints(_items, lower + 1, median );

		if (node->left != NULL) 
			node->left->parent = node;
		
		node->right = buildFromPoints(_items, median, upper );

		if (node->right != NULL) 
			node->right->parent = node;
	}

	return node;
}

// to avoid conflict with std::distance :/
float mydistance(struct Point *a,struct Point *b) {
  float d = 0.0;
  int i;
  for (i = 0; i < DIM; i++) {
    float diff = ((a -> coord)[i] - (b -> coord)[i]);
    d += (diff * diff);
  }
  return (sqrt(d));
}

void *search_entry(void *args)
{
	targs *ta = (targs *)args;
	struct Point *target;
	int i;

	dim3 blocks(NUM_OF_BLOCKS);
	dim3 tpb(NUM_OF_THREADS_PER_BLOCK);

	// added by Cambridge
	TIME_RESTART(extra);
	int _cam_npoints = (ta -> ub) - (ta -> lb);
	long nMatrixSize = _cam_npoints * DEPTH_STAT;
	printf("nsearchpoints = %d, DEPTH_STAT = %d, nMatrixSize = %d.\n", nsearchpoints, DEPTH_STAT, nMatrixSize);
//	SAFE_MALLOC(h_correlation_matrix, sizeof(int) * nMatrixSize);
	h_correlation_matrix = new int [nMatrixSize];
    CUDA_SAFE_CALL(cudaMalloc(&(d_correlation_matrix), sizeof(int) * nMatrixSize));

	cudaStream_t stream[nStreams];
	cudaEvent_t startEvent, stopEvent;
	float ms; // elapsed time in milliseconds
	CUDA_SAFE_CALL( cudaEventCreate(&startEvent) );
	CUDA_SAFE_CALL( cudaEventCreate(&stopEvent) );
	for (int i = 0; i < nStreams; ++i) {
		CUDA_SAFE_CALL( cudaStreamCreate(&stream[i]) );
	}

	printf("pre kernel start!\n");
//	TIME_START(pre_kernel);
	int stream_workload = nsearchpoints / nStreams;
	int point_offset = 0;
	int matrix_workload = nMatrixSize / nStreams;
    printf("nsearchpoints = %d, matrix_workload = %ld\n", nsearchpoints, matrix_workload);
	CUDA_SAFE_CALL( cudaEventRecord(startEvent,0) );
	for (int i = 0; i < nStreams; i ++) {
		search_pre_kernel<<<blocks, tpb, 0, stream[i]>>> (__the_tree_d, __GPU_point_Point_d, __GPU_point_array_d, __GPU_node_point_d, d_correlation_matrix, point_offset, point_offset + stream_workload, DEPTH_STAT, radius);
		point_offset += stream_workload;
	}
	for (int i = 0; i < nStreams; i ++) {
        printf("id: %f\n", matrix_workload * i);
		CUDA_SAFE_CALL(cudaMemcpyAsync(&h_correlation_matrix[matrix_workload * i], &d_correlation_matrix[matrix_workload * i], matrix_workload * sizeof(int), cudaMemcpyDeviceToHost, stream[i]));
	}
	CUDA_SAFE_CALL( cudaEventRecord(stopEvent, 0) );
	CUDA_SAFE_CALL( cudaEventSynchronize(stopEvent) );
	CUDA_SAFE_CALL( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
	printf("Time for asynchronous transfer and execute (ms): %f\n", ms);

	printf("pre kernel end!\n");

	TIME_START(CPU);
	int bytes = _cam_npoints * sizeof(int);
//	int* cpu_buffer;
	int* gpu_buffer;
	SAFE_MALLOC(cpu_buffer, bytes);
	memset(cpu_buffer, 0, bytes);
    CUDA_SAFE_CALL(cudaMalloc(&(gpu_buffer), bytes));

	vector<int> DATA;
	DATA.reserve(_cam_npoints);
	for (int id = 0; id < _cam_npoints; id ++)
	{
		DATA.push_back(id);
	}
	Sort(_cam_npoints, DATA, &cpu_buffer, 0, 0);
	delete [] h_correlation_matrix;
	cudaFree(d_correlation_matrix);
	TIME_END(CPU);

	CUDA_SAFE_CALL(cudaMemcpy(gpu_buffer, cpu_buffer, bytes, cudaMemcpyHostToDevice));

	TIME_END(extra);
	TIME_ELAPSED_PRINT(extra, stdout);

	TIME_START(kernel);
	search_kernel<<<blocks, tpb>>> (__the_tree_d, __GPU_point_Point_d, __GPU_point_array_d, __GPU_node_point_d, gpu_buffer, radius);
	cudaThreadSynchronize();		
	cudaError_t e = cudaGetLastError();
	if(e != cudaSuccess) {
		fprintf(stderr, "Error: search_kernel failed with error: %s\n", cudaGetErrorString(e));
		exit(1);
	}
	TIME_END(kernel);
	TIME_ELAPSED_PRINT(kernel, stdout);
	TIME_ELAPSED_PRINT(CPU, stdout);

  pthread_exit(0);
  return 0;
}

struct Point *read_point(FILE *in) {
#ifdef TRACK_TRAVERSALS
	static int id = 0;
#endif

	struct Point *p;
	SAFE_MALLOC(p, sizeof(struct Point));
		
	if(fscanf(in, "%d", &p->label) != 1) {
		fprintf(stderr, "Input file not large enough.\n");
		exit(1);
	}
	int j;
	for(j = 0; j < DIM; j++) {
		if(fscanf(in, "%f", &p->coord[j]) != 1) {
			fprintf(stderr, "Input file not large enough.\n");
			exit(1);
		}
	}

#ifdef TRACK_TRAVERSALS
	p->num_nodes_traversed = 0;
	p->num_trunc = 0;
	p->id = id++;
#endif

	p->corr = 0;
	return p;
}

struct Point *gen_point() {
#ifdef TRACK_TRAVERSALS
	static int id = 0;
#endif

	struct Point *p;
	SAFE_MALLOC(p, sizeof(struct Point));
	int j;
	p->label=0;
	for (j = 0; j < DIM; j++) {
		p->coord[j] = (float)rand() / RAND_MAX;
	}

#ifdef TRACK_TRAVERSALS
	p->num_nodes_traversed = 0;
	p->num_trunc = 0;
	p->id = id++;
#endif

	p->corr = 0;
	return p;
}

void read_input(int argc, char **argv, struct Point*** p_points, struct Point*** p_searchpoints) {
	
	int i, c;
	struct Point **points;
	struct Point **searchpoints;

	check_flag = 0;
	sort_flag = 0;
	verbose_flag = 0;
	nthreads = 1;
	i=0;
	while((c = getopt(argc, argv, "cvt:srw")) != -1) {
		switch(c) {
		case 'c':
			check_flag = 1;
			i++;
			break;

		case 'v':
			verbose_flag = 1;
			i++;
			break;

		case 't':
			nthreads = atoi(optarg);
			if(nthreads <= 0) {
				fprintf(stderr, "Error: invalid number of threads.\n");
				exit(1);
			}
			i+=2;
			break;

		case 's':
			sort_flag = 1;
			i++;
			break;
        
        case 'r':
            ratio_flag = 1;
            i ++;
            break;

        case 'w':
            warp_flag = 1;
            i ++;
            break;

		case '?':
			fprintf(stderr, "Error: unknown option.\n");
			exit(1);
			break;

		default:
			abort();
		}
	}
 
	if(argc - i < 2 || argc - i > 4) {
		fprintf(stderr, "usage: vptree [-c] [-v] [-t <nthreads>] [-s] <npoints> [input_file]\n");
		exit(1);
	}

	char *input_file = NULL;
	for(i = optind; i < argc; i++) {
		switch(i - optind) {
		case 0:
			input_file = argv[i];
			break;

		case 1:
			npoints = atoi(argv[i]);
            nsearchpoints = npoints;
			if(npoints <= 0) {
				fprintf(stderr, "Invalid number of points.\n");
				exit(1);
			}
			break;
        case 2:
            nsearchpoints = atoi(argv[i]);
            if (nsearchpoints <= 0) {
                fprintf(stderr, "Invalid number of points.\n");
                exit(1);
            }
            break;
		}
	}

	printf("Configuration: sort_flag = %d, verbose_flag = %d, nthreads=%d, DIM = %d, SPLICE_DEPTH = %d, nStreams = %d, npoints = %d, nsearchpoints = %d, input_file=%s\n", sort_flag, verbose_flag, nthreads, DIM, SPLICE_DEPTH, nStreams, npoints, nsearchpoints, input_file);

	// Allocate the point and search point arrays
	SAFE_CALLOC(points, npoints, sizeof(struct Point*));
	SAFE_CALLOC(searchpoints, nsearchpoints, sizeof(struct Point*));

	if (input_file != NULL) {
		FILE *in = fopen(input_file, "r");
		if( in == NULL) {
			fprintf(stderr, "Could not open %s\n", input_file);
			exit(1);
		}

		for (i = 0; i < npoints; i++) {
			points[i] = read_point(in);
		}

		for (i = 0; i < nsearchpoints; i++) {
//			searchpoints[i] = read_point(in);
			searchpoints[i] = points[i];
		}

		fclose(in);
	} else {
		for (i = 0; i < npoints; i++) {
			points[i] = gen_point();			
		}
		for (i = 0; i < nsearchpoints; i++) {
			searchpoints[i] = gen_point();
		}
	}
	
	*p_points = points;
	*p_searchpoints = searchpoints;
}

int main( int argc, char* argv[] ) {
	srand(0);
	int i;

	read_input(argc, argv, &points, &searchpoints);
	
	_items = points;
	_root = buildFromPoints(points, 0, npoints);	

	TIME_START(extra);
	check_depth(_root, 0);
	DEPTH_STAT --;
	TIME_END(extra);

	if(sort_flag) {
		buildFromPoints(searchpoints, 0, nsearchpoints);
	}

	TIME_START(init_kernel);
	init_kernel<<<1,1>>>();
	TIME_END(init_kernel);
	TIME_ELAPSED_PRINT(init_kernel, stdout);

	//print_tree(_root);

	int correct_cnt = 0;
//	int nsearchpoints = npoints; 	

	int rc;
	pthread_t * threads;
	SAFE_MALLOC(threads, sizeof(pthread_t)*nthreads);
	
	targs * args;
	SAFE_MALLOC(args, sizeof(targs)*nthreads);

	// Assign points to threads
	int start = 0;
	int j;
	for(j = 0; j < nthreads; j++) {
		int num = (nsearchpoints - start) / (nthreads - j);
		args[j].searchpoints = searchpoints;
		args[j].tid = j;
		args[j].lb = start;
		args[j].ub = start + num;
		start += num;
		//printf("%d %d\n", args[j].lb, args[j].ub);
	}

	TIME_START(search);

	__GPU_point_array_h = ((struct __GPU_point *)(malloc(sizeof(struct __GPU_point ) * (nsearchpoints))));
	if (__GPU_point_array_h == 0) {
		fprintf(stderr,"error [file=%s line=%d]: %s is NULL!","transformation",0,"__GPU_point_array_h");
		abort();
	}
	__GPU_point_Point_h = ((struct Point *)(malloc(sizeof(struct Point ) * (nsearchpoints))));
	if (__GPU_point_Point_h == 0) {
		fprintf(stderr,"error [file=%s line=%d]: %s is NULL!","transformation",0,"__GPU_point_Point_h");
		abort();
	}
	__GPU_node_point_h = ((struct Point *)(malloc(sizeof(struct Point ) * (nsearchpoints))));
	if (__GPU_node_point_h == 0) {
		fprintf(stderr,"error [file=%s line=%d]: %s is NULL!","transformation",0,"__GPU_node_point_h");
		abort();
	}

	struct Point *target;
	for (int i = 0; i < nsearchpoints; i++) {
		target = searchpoints[i];
		target -> tau = 3.40282347e+38F;
		memcpy(__GPU_point_Point_h + i,target,sizeof(struct Point ) * 1);
		__GPU_point_array_h[i].target = i;
	}

	__the_tree_h = __GPU_buildTree(_root,npoints);

	__the_tree_d = __GPU_allocDeviceTree(__the_tree_h);
	__GPU_memcpyTreeToDev(__the_tree_h,__the_tree_d);

	TIME_START(search_kernel);
	for(i = 0; i < nthreads; i++) {		
		rc = pthread_create(&threads[i], NULL, search_entry, &args[i]);
		if(rc) {
			fprintf(stderr, "Error: could not create thread, rc = %d\n", rc);
			exit(1);
		}
	}
	
	// wait for threads
	for(i = 0; i < nthreads; i++) {
		pthread_join(threads[i], NULL);
	}

	TIME_END(search_kernel);
	TIME_ELAPSED_PRINT(search_kernel, stdout);

	__GPU_memcpyTreeToHost(__the_tree_h, __the_tree_d);

	for (i = 0; i < nsearchpoints; i++) {
		target = searchpoints[i];
		memcpy(target, __GPU_point_Point_h + i,sizeof(struct Point ) * 1);
	}

	__GPU_freeDeviceTree(__the_tree_d);

	// compute correct count
	for (i = 0; i < nsearchpoints; i++) {
		struct Point *target = searchpoints[i];
		corr_sum += target->corr;
	}
	
	TIME_END(search);
	TIME_ELAPSED_PRINT(search, stdout);

#ifdef TRACK_TRAVERSALS
	unsigned long long sum_nodes_traversed = 0;
	long sum_trunc = 0;
	int na;
	long maximum = 0, all = 0;
    int num_of_warps = 0;
	unsigned long long maximum_sum = 0, all_sum = 0;
	for(i = 0; i < nsearchpoints + (nsearchpoints % 32); i+=32) {
//		struct Point* p = searchpoints[i];
		struct Point* p = searchpoints[cpu_buffer[i]];
		sum_nodes_traversed += p->num_nodes_traversed;
		sum_trunc += p->num_trunc;
		na = p->num_nodes_traversed;
   
        for(j = i + 1; j < i + 32 && j < nsearchpoints; j++) {
//          p = searchpoints[j];
            p = searchpoints[cpu_buffer[j]];
            if(p->num_nodes_traversed > na)
               na = p->num_nodes_traversed;
            sum_nodes_traversed += p->num_nodes_traversed;
            sum_trunc += p->num_trunc;

        }

        if (warp_flag) {
            p = searchpoints[cpu_buffer[i]];
		    maximum = p->num_nodes_traversed;
	    	all = p->num_nodes_traversed;
    		for(j = i + 1; j < i + 32 && j < nsearchpoints; j++) {
//			    p = searchpoints[j];
			    p = searchpoints[cpu_buffer[j]];

			    if (p->num_nodes_traversed > maximum)
				    maximum = p->num_nodes_traversed;
			    all += p->num_nodes_traversed;
		    }
//		    printf("nodes warp %d: %d\n", i/32, na);
		    printf("%d\n", maximum);
		    maximum_sum += maximum;
		    all_sum += all;
            num_of_warps ++;
        }
	}
    printf("avg num of traversed nodes per warp is %f\n", (float)maximum_sum / num_of_warps);
	printf("@ maximum_sum: %llu\n", maximum_sum);
	printf("@ all_sum: %llu\n", all_sum);

	printf("@ sum_nodes_traversed: %llu\n", sum_nodes_traversed);
	printf("@ avg_nodes_traversed: %f\n", (float)sum_nodes_traversed / nsearchpoints);
	printf("sum_trunc:%d\n", sum_trunc);
#endif

	printf("Full correlation: %d， nsearchpoints = %d\n", corr_sum, nsearchpoints);
	printf("avg corr: %f\n", (float)corr_sum / nsearchpoints);

	// TODO: free the rest but its not really important
	/*
	for(i = 0; i < npoints; i++) {
		free(points[i]);
		free(searchpoints[i]);
	}
	free(points);
	free(searchpoints);
	*/
	return 0;
}

void check_depth(Node * node, int _depth)
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
