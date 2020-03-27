/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include "nn_functions.h"

static int sort_split;

void read_input(int argc, char **argv) 
{
	unsigned long long i, j, k,c;
	float min = FLT_MAX;
	float max = FLT_MIN;
	FILE *in;
	char *input_file;

	if(argc < 2) {
		fprintf(stderr, "usage: nn [-c] [-v] [-t <nthreads>] [-s] <input_file> <npoints> [<nsearchpoints>], %d\n", argc);
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

		case 't':
			nthreads = atoi(optarg);
			if(nthreads <= 0) {
				fprintf(stderr, "Error: invalid number of threads.\n");
				exit(1);
			}
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
	
	for(i = optind; i < argc; i++) 
	{
		switch(i - optind) 
		{
		case 0:
			input_file = argv[i];
			break;

		case 1:
				npoints = atoi(argv[i]);
				nsearchpoints = npoints;
				if(npoints <= 0) 
				{
					fprintf(stderr, "Not enough points.\n");
					exit(1);
				}
				break;

		case 2:
			nsearchpoints = atoi(argv[i]);
			if(nsearchpoints <= 0) 
			{
				fprintf(stderr, "Not enough search points.");
				exit(1);
			}
			break;
		}
	}

    training_points = alloc_points(npoints);
    search_points = alloc_points(nsearchpoints);

	if(strcmp(input_file, "random") == 0) 
	{
		srand(0);
		for(i = 0; i < npoints; i++) 
		{
			training_points[i].id = i;
			for(j = 0; j < DIM; j++) 
			{
				training_points[i].coord[j] = 1.0 + (float)rand() / RAND_MAX;			
			}
		}

		for(i = 0; i < nsearchpoints; i++) 
		{
			search_points[i].id = i;
			for(j = 0; j < DIM; j++) 
			{
				search_points[i].coord[j] = 1.0 + (float)rand() / RAND_MAX;			
			}
		}
	} 
	else 
	{
		in = fopen(input_file, "r");
		if(in == NULL) 
		{
			fprintf(stderr, "Could not open %s\n", input_file);
			exit(1);
		}

		for(i = 0; i < npoints; i++) 
		{
			read_point(in, &training_points[i]);
            training_points[i].id = i;
		}

		for(i = 0; i < nsearchpoints; i++) 
		{
			read_point(in, &search_points[i]);
            search_points[i].id = i;
		}
		fclose(in);
	}
}

void read_point(FILE *in, Point *p) 
{
	int j;
	if(fscanf(in, "%d", &p->label) != 1) 
	{
		fprintf(stderr, "Input file not large enough.\n");
		exit(1);
	}
	for(j = 0; j < DIM; j++) 
	{
		if(fscanf(in, "%f", &p->coord[j]) != 1) 
		{
			fprintf(stderr, "Input file not large enough.\n");
			exit(1);
		}
	}
}

int can_correlate(Point * point, KDCell * cell, float rad) 
{
	float dist=0.0;
	float boxdist=0.0;
	float sum=0.0;
	float boxsum=0.0;
	float center=0.0;
	int i;

	for(i = 0; i < DIM; i++) 
	{
		center = (cell->max[i] + cell->min[i]) / 2;
		boxdist = (cell->max[i] - cell->min[i]) / 2;
		dist = point->coord[i] - center;
		sum += dist * dist;
		boxsum += boxdist * boxdist;
	}

	if(sqrt(sum) - sqrt(boxsum) < sqrt(rad))
		return 1;
	else
		return 0;
}

static inline float distance_axis(Point *a, Point *b, int axis) 
{
	return (a->coord[axis] - b->coord[axis]) * (a->coord[axis] - b->coord[axis]);
}

static inline float distance(Point *a, Point *b) 
{
	int i;
	float d = 0;
	for(i = 0; i < DIM; i++) 
	{
		d += distance_axis(a,b,i);		
	}
	return d;
}

int compare_point(const void *a, const void *b) {
	if(((struct Point *)a)->coord[sort_split] < ((struct Point *)b)->coord[sort_split]) {
		return -1;
	} else if(((struct Point *)a)->coord[sort_split] > ((struct Point *)b)->coord[sort_split]) {
		return 1;
	} else {
		return 0;
	}
}


float mMax(float a, float b) 
{
	return a > b ? a : b;
}

float mMin(float a, float b) 
{
	return a < b ? a : b;
}

KDCell * construct_tree(Point * points, int lb, int ub, int depth, int index) 
{
	KDCell *node = alloc_kdcell();
    node->id = index;

	int size = ub - lb + 1;
	int mid;
	int i, j;

	if (size <= MAX_POINTS_IN_CELL) 
	{
		for (i = 0; i < size; i++) 
		{
			node->points[i] = lb + i;
			for (j = 0; j < DIM; j++) 
			{
				node->max[j] = mMax(node->max[j], points[lb + i].coord[j]);
				node->min[j] = mMin(node->min[j], points[lb + i].coord[j]);
				//printf("%d: %f %f %f\n", index, node->max[j], node->min[j], points[lb + i].coord[j]);
			}
			//exit(0);
		}
		node->axis = DIM; // leaf node has axis of DIM
		return node;
	} 
	else 
	{
		sort_split = depth % DIM;
		qsort(&points[lb], ub - lb + 1, sizeof(struct Point), compare_point);
		mid = (ub + lb) / 2;

		node->axis = depth % DIM;
		node->splitval = points[mid].coord[node->axis];
		node->left = construct_tree(points, lb, mid, depth + 1, 2 * index);
		node->right = construct_tree(points, mid+1, ub, depth + 1, 2 * index + 1);

		for(j = 0; j < DIM; j++) 
		{
			node->min[j] = mMin(node->left->min[j], node->right->min[j]);
			node->max[j] = mMax(node->left->max[j], node->right->max[j]);
			//printf("%f %f %f\n", node->max[j], node->min[j], node->left->min[j]);
		}
		return node;
	}	
}

void sort_points(Point * points, int lb, int ub, int depth) 
{
	int mid;
	if(lb >= ub)
		return;

	sort_split = depth % DIM;
	qsort(&points[lb], ub - lb + 1, sizeof(Point), compare_point);
	mid = (ub + lb) / 2;

	if(mid > lb) 
	{
		sort_points(points, lb, mid - 1, depth + 1);
		sort_points(points, mid, ub, depth + 1);
	} 
	else 
	{
		sort_points(points, lb, mid, depth + 1);
		sort_points(points, mid+1, ub, depth + 1);
	}
}

void update_closest(Point *point, int candidate_index)
{
    float dist = distance(point, &training_points[candidate_index]);
    Point *tmppt;
    float tmpdist;
    int n;

    if (dist > point->closest_dist)
    {
        return;
    }

    point->closest = candidate_index;
    point->closest_dist = dist;
}

void free_tree(KDCell *n)
{
	if (n->left != NULL) free_tree(n->left);
	if (n->right != NULL) free_tree(n->right);
	delete n;
}

Point* alloc_points(int n)
{
    int i, j;
    Point *points;
    SAFE_MALLOC(points, sizeof(Point) * n);
    for (i = 0; i < n; i++)
    {
        points[i].closest_dist = FLT_MAX;
        points[i].closest = -1;
        for(j = 0; j < DIM; j ++)
        {
            points[i].coord[j] = 0.0f;
        }
        points[i].id = -1;
        points[i].label = -1;
        #ifdef TRACK_TRAVERSALS
        points[i].num_nodes_traversed = 0;
        #endif
    }
    return points;
}

KDCell* alloc_kdcell()
{
    int i;
    KDCell *cell = (KDCell*) malloc(sizeof(KDCell));

    for (i = 0; i < DIM; i++)
    {
        cell->min[i] = FLT_MAX;
        cell->max[i] = FLT_MIN;
    }

    for (i = 0; i < MAX_POINTS_IN_CELL; i++)
    {
        cell->points[i] = -1;
    }

    cell->left = NULL;
    cell->right = NULL;
    cell->axis = -1;
    cell->splitval = FLT_MAX;
    cell->id = -1;
    return cell;
}

