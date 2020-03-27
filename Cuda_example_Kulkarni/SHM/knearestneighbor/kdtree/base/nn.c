/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <unistd.h>
#include <time.h>
#include <getopt.h>
#include <pthread.h>
#include "util_common.h"
#include "nn.h"

node *points;
node *search_points;
unsigned int npoints;
unsigned int nsearchpoints;

float* nearest_distance;
int* nearest_point_index;

float* nearest_distance_brute;
int* nearest_point_index_brute;

#ifdef TRACK_TRAVERSALS
int *nodes_accessed;
#endif

node * tree;

int K;
int nthreads = 1;
char *input_file;
int sort_flag = 0;
int verbose_flag = 0;
int check_flag = 0;

static void* thread_function(void *arg);

static inline float distance(float *a, float *b);
TIME_INIT(runtime);
TIME_INIT(construct_tree);
TIME_INIT(traversal_time);

int main(int argc, char **argv) {
	int i, j, k;
	
	struct thread_args *args;
	pthread_t *threads;

	TIME_START(runtime);

	read_input(argc, argv);
	TIME_START(construct_tree);

	if(sort_flag) {
		//sort_search_points(search_points, 0, nsearchpoints - 1, 0);
		construct_tree(search_points, 0, nsearchpoints - 1, 0);
	}

	#ifdef TRACK_TRAVERSALS
	SAFE_MALLOC(nodes_accessed, sizeof(int)*nsearchpoints);
	#endif
	
	tree = construct_tree(points, 0, npoints - 1, 0);
	
	TIME_END(construct_tree);
	TIME_START(traversal_time);
	SAFE_MALLOC(args, sizeof(struct thread_args)*nthreads);
	SAFE_MALLOC(threads, sizeof(pthread_t)*nthreads);

	// Assign points to threads
	int start = 0;
	for(j = 0; j < nthreads; j++) {
		int num = (nsearchpoints - start) / (nthreads - j);
		args[j].tid = j;
		args[j].lb = start;
		args[j].ub = start + num;
		start += num;

		//printf("%d %d\n", args[j].lb, args[j].ub);
	}
	
	for(j = 0; j < nthreads; j++) {
		if(pthread_create(&threads[j], NULL, thread_function, &args[j]) != 0) {
			fprintf(stderr, "Could not create thread %d\n", j);
			exit(1);
		}
	}

	for(j = 0; j < nthreads; j++) {
		pthread_join(threads[j], NULL);
	}
	
	free(args);
	free(threads);


	#ifdef TRACK_TRAVERSALS
	long int sum_nodes_accessed=0;
	for(i = 0; i < nsearchpoints; i++) {
		sum_nodes_accessed += (long int)nodes_accessed[i];
	}
	#endif
	TIME_END(traversal_time);
	TIME_END(runtime);

	if(verbose_flag) {
		for(j = 0; j < nsearchpoints; j++) {
			printf("\n%d: ", j);
			for(i = 0; i < K; i++) {
				if(i == K-1)
					printf("%d (%1.3f)", nearest_point_index[j*K + i], nearest_distance[j*K+i]);
				else
					printf("%d (%1.3f),", nearest_point_index[j*K + i], nearest_distance[j*K+i]);
			}

			if(check_flag) {
				printf(" :: ");
				for(i = 0; i < K; i++) {
					if(i == K-1)
						printf("%d", nearest_point_index_brute[j*K + i]);
					else
						printf("%d,", nearest_point_index_brute[j*K + i]);
				}
			}
		}
		printf("\n");
	}

	if(check_flag) {
		int found;
		for(j = 0; j < nsearchpoints; j++) {
			found = 0;
			for(i = 0; i < K; i++) {				
				for(k = 0; k < K; k++) {
					if(nearest_point_index[j*K + i] == nearest_point_index_brute[j*K + k]) {
						found++;
						break;
					}
				}
			}
			if(found != K) {
				printf("ERROR: Invalid Results Detected, %d\n", found);
				break;
			}
		}
	}

	TIME_ELAPSED(traversal_time);
	#ifdef TRACK_TRAVERSALS
	printf("sum nodes accessed: %llu\n", sum_nodes_accessed);
	printf("avg nodes accessed: %f\n", (float)sum_nodes_accessed/nsearchpoints);
	#endif
	/*TIME_ELAPSED_PRINT(construct_tree, stdout);
	TIME_ELAPSED_PRINT(traversal_time, stdout);
  	TIME_ELAPSED_PRINT(runtime, stdout);*/
	printf("Traversal time %f seconds\n", traversal_time_elapsed/1000);
	return 0;
}

void* thread_function(void *arg) {
	
	struct thread_args *args = (struct thread_args*)arg;
	int j, i;
	for(j = args->lb; j < args->ub; j++) {
		for(i = 0; i < K; i++) {
			nearest_distance[K*j + i] = FLT_MAX;
			nearest_point_index[K*j + i] = -1;			
		}
		
		#ifdef TRACK_TRAVERSALS
		nodes_accessed[j] = 0;
		#endif

		nearest_neighbor_search(&search_points[j], tree, j, -FLT_MAX);	 		
	}

	if(check_flag) {
		for(j = args->lb; j < args->ub; j++) {
			for(i = 0; i < K; i++) {
				nearest_distance_brute[K*j + i] = FLT_MAX;
				nearest_point_index_brute[K*j + i] = -1;			
			}

			nearest_neighbor_search_brute(&search_points[j], j);
		}
	}
}

static inline void update_closest(float *nearest_dist, int *nearest_idx, float dist, int idx, int pidx) {
	float *ptrdist = &nearest_dist[pidx*K];
	int *ptridx = &nearest_idx[pidx*K];
	int n, ch;
	int tmpidx; 
	float tmpdist;
	int r;

	if (dist > ptrdist[0]) {
		return;
	}

	ptrdist[0] = dist;
	ptridx[0] = idx;
	
	for(n = 1; n < K && ptrdist[n-1] < ptrdist[n]; n++) {
		tmpidx = ptridx[n];
		tmpdist = ptrdist[n];
		ptridx[n] = ptridx[n-1];
		ptrdist[n] = ptrdist[n-1];
		ptridx[n-1] = tmpidx;
		ptrdist[n-1] = tmpdist;
	} 
}

void nearest_neighbor_search(node *point, node *current_node, int pidx, float axis_dist) {
	int i;
	int near_child = -1; // -1 for none, 0 for left, 1 for right
	int current_split;
	int max_neighbor_idx = 0;
	float max_neighbor_dist;

	// is this node closer than the current best?
	float dist;
	
	if(axis_dist > nearest_distance[pidx*K])
		return;

	#ifdef TRACK_TRAVERSALS
	nodes_accessed[pidx]++;
	#endif

	current_split = current_node->axis;

	dist = distance(current_node->point, point->point);
	update_closest(nearest_distance, nearest_point_index, dist, current_node->point_index, pidx);
	
	if(current_split != DIM) {

		axis_dist = distance_axis(point->point, current_node->point, current_split);

		if(current_node->left != NULL && point->point[current_split] <= current_node->point[current_split]) {
			nearest_neighbor_search(point, current_node->left, pidx, -FLT_MAX);
			if(current_node->right != NULL)
				nearest_neighbor_search(point, current_node->right, pidx, axis_dist);
		} else if(current_node->right != NULL) {
			nearest_neighbor_search(point, current_node->right, pidx, -FLT_MAX);
			if(current_node->left != NULL)
				nearest_neighbor_search(point, current_node->left, pidx, axis_dist);
		}
	}
}

void nearest_neighbor_search_brute(node *point, int pidx) {
	int i;
	float dist;
	for(i = 0; i < npoints; i++) {
		dist = distance(points[i].point, point->point);
		update_closest(nearest_distance_brute, nearest_point_index_brute, dist, points[i].point_index, pidx);
	}
}

static inline float distance(float *a, float *b) {
	int i;
	float d = 0;
	// returns distance squared
	for(i = 0; i < DIM; i++) {
		d += distance_axis(a,b,i);		
	}

	return d;
}

void read_point(node *p, FILE *in, int index, int random) {
	int junk;
	float data;
	int j;

	if(random) {
		for(j = 0; j < DIM; j++) {
			p->point[j] = 1.0 + (float)rand() / RAND_MAX;
		}
	} else {
		/*if(fscanf(in, "%d", &junk) != 1) {
			fprintf(stderr, "Input file not large enough.\n");
			exit(1);
		}*/
		for(j = 0; j < DIM; j++) {
			if(fscanf(in, "%f", &data) != 1) {
				fprintf(stderr, "Input file not large enough.\n");
				exit(1);
			}
			p->point[j] = data;
		}
	}
}

void read_input(int argc, char **argv) {
	int i, j,c;

	check_flag = 0;
	sort_flag = 0;
	verbose_flag = 0;
	i=0;
	while((c = getopt(argc, argv, "cvt:s")) != -1) {
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

		case '?':
			fprintf(stderr, "Error: unknown option.\n");
			exit(1);
			break;

		default:
			abort();
		}
	}
 
	if(argc - i < 4 || argc - i > 5) {
		fprintf(stderr, "usage: nn [-c] [-v] [-t <nthreads>] [-s] <k> <input_file> <npoints> [<nsearchpoints>]\n");
		exit(1);
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

	printf("configuration: sort_flag=%d, check_flag = %d, verbose_flag=%d, nthreads=%d, K=%d, input_file=%s, npoints=%d, nsearchpoints=%d\n", sort_flag, check_flag, verbose_flag, nthreads, K, input_file, npoints, nsearchpoints);
	
	SAFE_CALLOC(points, npoints, sizeof(node));
    	SAFE_CALLOC(search_points, nsearchpoints, sizeof(node));
	SAFE_MALLOC(nearest_distance, sizeof(float)*nsearchpoints*K);
	SAFE_MALLOC(nearest_point_index, sizeof(int)*nsearchpoints*K);

	if(check_flag) {
		SAFE_MALLOC(nearest_distance_brute, sizeof(float)*nsearchpoints*K);
		SAFE_MALLOC(nearest_point_index_brute, sizeof(int)*nsearchpoints*K);
	}
	if(strcmp(input_file, "random") != 0) {
		FILE * in = fopen(input_file, "r");
		if(in == NULL) {
			fprintf(stderr, "Could not open %s\n", input_file);
			exit(1);
		}

		for(i = 0; i < npoints; i++) {
			read_point(&points[i], in, i, 0);
		}
	
		for(i = 0; i < nsearchpoints; i++) {
			read_point(&search_points[i], in, i, 0);
		}
		
		fclose(in);
		
	} else {
		for(i = 0; i < npoints; i++) {
			read_point(&points[i], NULL, i, 1);
		}
	
		for(i = 0; i < nsearchpoints; i++) {
			read_point(&search_points[i], NULL, i, 1);
		}
	}
}
