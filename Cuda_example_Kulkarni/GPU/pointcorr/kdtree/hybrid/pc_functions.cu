/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include "../../../common/util_common.h"
#include "pc_functions.h"

void read_input(int argc, char *argv[]) 
{
	unsigned long long i = 0;
	unsigned long long j = 0;
	unsigned long c = 0;
	int label = 0;
	FILE *in;
	char *input_file;

	if(argc < 2) 
	{
		fprintf(stderr, "usage: point_corr <npoints> [input_file]\n");
		exit(1);
	}

	while((c = getopt(argc, argv, "cvt:srw")) != -1)
	{
		switch(c)
		{
			case 'c':
				check_flag = 1;
				break;

			case 'v':
				verbose_flag = 1;
				break;

			case 't':
				nthreads = atoi(optarg);
				if(nthreads <= 0) 
				{
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
				if(npoints <= 0) 
				{
					fprintf(stderr, "Not enough points.\n");
					exit(1);
				}
				break;
			
			default:
				break;
		}
	}

	// initialize the points	
	SAFE_MALLOC(points, sizeof(kd_cell*)*npoints);
	
	for(i = 0; i < npoints; i ++) 
	{
		SAFE_MALLOC(points[i], sizeof(kd_cell));
		points[i]->splitType = SPLIT_LEAF;
		points[i]->left = NULL;
		points[i]->right = NULL;
		points[i]->corr = 0;
		#ifdef TRACK_TRAVERSALS
		points[i]->nodes_accessed = 0;
		points[i]->nodes_truncated = 0;
		#endif
		points[i]->id = i;
		for(j = 0; j < DIM; j ++) 
		{
//			points[i]->coord_max[j] = -1000000.0;
			points[i]->coord_max[j] = FLT_MIN;
			points[i]->min[j] = FLT_MAX;
		}
	}
	
	if(strcmp(input_file, "random") != 0) 
	{
		in = fopen(input_file, "r");
		if(in == NULL) 
		{
			fprintf(stderr, "Could not open %s\n", input_file);
			exit(1);
		}		

		for(i = 0; i < npoints; i++) 
		{
			if(fscanf(in, "%d", &label) != 1)
			{
				fprintf(stderr, "Input file not large enough.\n");
				exit(1);
			}
			for(j = 0; j < DIM; j++) 
			{
				if(fscanf(in, "%f", &(points[i]->coord_max[j])) != 1) 
				{
					fprintf(stderr, "Error: Invalid point %d\n", i);
					exit(1);
				}
				points[i]->min[j] = points[i]->coord_max[j];
			}
			points[i]->id = i;
		}
		if(in != stdin) 
		{
			fclose(in);
		}
	}
	else 
	{
		srand(0);
		for(i = 0; i < npoints; i++)
		{
			points[i]->id = i;
			for(j = 0; j < DIM; j++) 
			{
				points[i]->coord_max[j] = (float)rand() / RAND_MAX;
				points[i]->min[j] = points[i]->coord_max[j];
			}
		}
	}

}

kd_cell * build_tree(kd_cell ** points, int split, int lb, int ub, int index)
{
	int mid = 0;
	int i = 0;
	kd_cell *node = NULL;
	sum_of_nodes ++;	

	if(lb > ub)
		return 0;
	if(lb == ub)
	{
//		return points[lb];
		SAFE_MALLOC(node, sizeof(kd_cell));
		node->splitType = SPLIT_LEAF;
#ifdef TRACK_TRAVERSALS
		node->nodes_accessed = 0;
		node->nodes_truncated = 0;
#endif
		node->id = index;
		for(i = 0; i < DIM; i ++)
		{
			node->coord_max[i] = points[lb]->coord_max[i];
			node->min[i] = points[lb]->min[i];
		}
		node->corr = 0;
		node->left = NULL;
		node->right = NULL;
		return node;
	}
	else
	{
		sort_split = split;
		qsort(&points[lb], ub - lb + 1, sizeof(kd_cell*), kdnode_cmp);
		mid = (ub + lb) / 2;
		
		// create a new node to contains the points:
		SAFE_MALLOC(node, sizeof(kd_cell));
		node->splitType = split;
#ifdef TRACK_TRAVERSALS
		node->nodes_accessed = 0;
		node->nodes_truncated = 0;
#endif
		node->id = index;
		node->left = build_tree(points, (split + 1) % DIM, lb, mid, 2 * index);
		node->right = build_tree(points, (split + 1) % DIM, mid + 1, ub, 2 * index + 1);
		node->corr = 0;
		
		for(i = 0; i < DIM; i ++)
		{
			node->min[i] = FLT_MAX;
//			node->coord_max[i] = -1000000.0;
			node->coord_max[i] = FLT_MIN;
		}

		if(node->left != NULL) 
		{
			for(i = 0; i < DIM; i++) 
			{
				if(node->coord_max[i] < node->left->coord_max[i]) 
				{
					node->coord_max[i] = node->left->coord_max[i];
				}

				if(node->min[i] > node->left->min[i]) 
				{
					node->min[i] = node->left->min[i];
				}
			}
		}

		if(node->right != NULL) 
		{
			for(i = 0; i < DIM; i++) 
			{
				if(node->coord_max[i] < node->right->coord_max[i]) 
				{
					node->coord_max[i] = node->right->coord_max[i];
				}
				if(node->min[i] > node->right->min[i]) 
				{
					node->min[i] = node->right->min[i];
				}
			}
		}

		return node;
	}
}

int kdnode_cmp(const void *a, const void *b)
{
	/* note: sort split must be updated before call to qsort */
	kd_cell **na = (kd_cell**)a;
	kd_cell **nb = (kd_cell**)b;
	
	if((*na)->coord_max[sort_split] < (*nb)->coord_max[sort_split])
	{
		return -1;
	}
	else if((*na)->coord_max[sort_split] > (*nb)->coord_max[sort_split])
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

void free_tree(kd_cell *root)
{
	if(root->left != NULL)
		free_tree(root->left);
	if(root->right != NULL)
		free_tree(root->right);
	delete root;
}

static inline float distance_axis(kd_cell *a, kd_cell *b, int axis)
{
    return (a->coord_max[axis] - b->coord_max[axis]) * (a->coord_max[axis] - b->coord_max[axis]);
}

static inline float distance(kd_cell *a, kd_cell *b)
{
    float d = 0;
    int i = 0;
    for (i = 0; i < DIM; i ++)
    {
        d += distance_axis(a, b, i);
    }
    return d;
}

bool can_correlate(kd_cell *point, kd_cell *cell)
{
    float sum = 0.0f;
    float boxsum = 0.0f;
    int i = 0;
    for (i = 0; i < DIM; i ++)
    {
        float center = (cell->coord_max[i] + cell->min[i]) / 2;
        float boxdist = (cell->coord_max[i] - cell->min[i]) / 2;
        float dist = point->coord_max[i] - center;
        sum += dist * dist;
        boxsum += boxdist * boxdist;
	}

    if ((sqrt(sum) - sqrt(boxsum)) < RADIUS)
        return true;
    else
        return false;
}

