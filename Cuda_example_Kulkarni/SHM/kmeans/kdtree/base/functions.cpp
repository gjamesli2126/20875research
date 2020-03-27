/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/

#include "functions.hpp"


void read_input(int argc, char **argv) {
    int i, j;
    int c;
    char* input_file = NULL;
    
    if(argc < 2) {
        fprintf(stderr, "usage: kmeans [-t] <DIM> <k> <input_file>/\"random\" <npoints>\n");
        exit(1);
    }
    
    while((c = getopt(argc, argv, "t")) != -1) {
        switch(c) {
            case 't':
                nthreads = atoi(optarg);
                if(nthreads <= 0) {
                    fprintf(stderr, "Error: invalid number of threads.\n");
                    exit(1);
                }
                i+=2;
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
		DIM = atoi(argv[i]);
		if(DIM <=0)
                {	
		    fprintf(stderr, "Invalid number of dimensions.\n");
                    exit(1);
		}
		break;
            case 1:
                K = atoi(argv[i]);
                if(K <= 0) {
                    fprintf(stderr, "Invalid number of clusters.\n");
                    exit(1);
                }
                break;
                
            case 2:
                input_file = argv[i];
                break;
                
            case 3:
                npoints = atoi(argv[i]);
                if(npoints <= 0) {
                    fprintf(stderr, "Not enough points.\n");
                    exit(1);
                }
                break;
	    default: 
		break;
        }
    }
    
    printf("configuration: DIM=%d nthreads=%d K=%d, input_file=%s, npoints=%d.\n", DIM, nthreads, K, input_file, npoints);
    
    SAFE_CALLOC(points, npoints, sizeof(DataPoint));
    SAFE_CALLOC(clusters, K, sizeof(ClusterPoint));
    
    char junkc[250];
    if(strcmp(input_file, "random") != 0) {
        FILE * in = fopen(input_file, "r");
        if(in == NULL) {
            fprintf(stderr, "Could not open %s\n", input_file);
            exit(1);
        }
        
        int junk;
        float data;
        
        for(i = 0; i < npoints; i++) {
            points[i].idx = i;
	    memset(junkc,0,250);
            if(fscanf(in, "%s", &junkc) != 1) {
                fprintf(stderr, "Input file not large enough.\n");
                exit(1);
            }
	    points[i].coord = new float[DIM];
	    points[i].label = new char[strlen(junkc)+1];
            strcpy(points[i].label, junkc);
            for(j = 0; j < DIM; j++) {
                if(fscanf(in, "%f,", &data) != 1) {
                    fprintf(stderr, "Input file not large enough.\n");
                    exit(1);
                }
                points[i].coord[j] = data;
                points[i].clusterId = -1;
            }
        }
        fclose(in);
    } else {
        for(i = 0; i < npoints; i++) {
	    memset(junkc,0,250);
            points[i].idx = i;
            sprintf(junkc,"%d",rand() % DIM);
	    points[i].coord = new float[DIM];
	    points[i].label = new char[strlen(junkc)+1];
	    strcpy(points[i].label,junkc);
            for(j = 0; j < DIM; j++) {
                points[i].coord[j] = 1.0 + (float)rand() / RAND_MAX;
            }
            points[i].clusterId = -1;
        }
    }
}

Node * construct_tree(ClusterPoint *clusters, int start_idx, int end_idx, int depth, Node* parent) {
	int i;

	if (start_idx > end_idx)
		return NULL;
    	if (depth > max_depth)
        	max_depth = depth;

	if (start_idx == end_idx) {
		Node* node = new Node();
        nnodes ++;
		node->pivot = &clusters[start_idx];
        node->parent = parent;
		node->axis = DIM;
        node->depth = depth;
		node->left = NULL;
		node->right = NULL;
		return node;
	} else {
		// this is not a single point:
		int split = depth % DIM;

		// find the median and partition clusters into left and right sets:
		int median_index;
		int j;

		// partition to get some sort of split around the median
		sort_split = split;
		qsort(&clusters[start_idx], end_idx - start_idx + 1, sizeof(ClusterPoint), cmpfn_float);
		median_index = (start_idx + end_idx) / 2;

		Node* node = new Node();
#ifdef METRICS
	if(depth == (splice_depth-1))
		subtrees.push_back(node);
#endif

        nnodes ++;
        node->pivot = &clusters[median_index];
        node->parent = parent;
		node->axis = split;
        node->depth = depth;
		node->left = construct_tree(clusters, start_idx, median_index - 1, depth + 1, node);
		node->right = construct_tree(clusters, median_index + 1, end_idx, depth + 1, node);
		return node;
	}
}

void deconstruct_tree(Node* root) {
    if (root->left != NULL) 
        deconstruct_tree(root->left);
    if (root->right != NULL)
        deconstruct_tree(root->right);

    nnodes --;
    delete root;
}

static int cmpfn_float(const void *a, const void *b) {
    ClusterPoint* pa = (ClusterPoint*) a;
    ClusterPoint* pb = (ClusterPoint*) b;
    
    if (pa->pt.coord[sort_split] < pb->pt.coord[sort_split])
        return -1;
    else if (pa->pt.coord[sort_split] == pb->pt.coord[sort_split])
        return 0;
    else
        return 1;
}

void PrintClusters(FILE* fp) {
	for (int i = 0; i < K; i ++) {
        fprintf(fp,"%dth cluster has %d points, coord: ", i, clusters[i].num_of_points);
        for (int j = 0; j < DIM; j ++) {
            fprintf(fp,"%f, ", clusters[i].pt.coord[j]);
        }
        fprintf(fp,"\n");
    }
}

#ifdef METRICS
void printLoadDistribution(bool iterative)
{
	if(!iterative)
		printf("num bottom subtrees %d\n",subtrees.size());
	std::vector<Node*>::iterator iter = subtrees.begin();
	for(;iter != subtrees.end();iter++)
	{
		long int num_vertices=0, footprint=0;
		subtreeStats stats;
		getSubtreeStats(*iter, &stats);
		if(!iterative)
			printf("id (%p) num_vertices %d footprint %ld\n", *iter, stats.numnodes, stats.footprint);
		else
			subtreeStatsList.push_back(stats);
	}
}

void getSubtreeStats(Node* ver, subtreeStats* stats)
{
		stats->numnodes += 1;
		stats->footprint += ver->numpointsvisited;
		assert(ver != NULL);

		if((ver->left == NULL) && (ver->right==NULL))
		{
			return;
		}

		if(ver->left)
		{
			getSubtreeStats(ver->left,stats);
		}
		if(ver->right)
		{
			getSubtreeStats(ver->right,stats);
		}
}

#endif
