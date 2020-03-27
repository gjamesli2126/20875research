/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include "functions.hpp"

void read_input(int argc, char **argv, int procRank, int numProcs) {
    int i, j,k;
    int c;
    char* input_file = NULL;
    
    if(argc < 2) {
        fprintf(stderr, "usage: nn [-c] [-v] [-s] <DIM> <k> <input_file> <npoints>\n");
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
                i+=2;
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
    if(procRank == 0)
    	printf("configuration: sort_flag=%d, check_flag = %d, verbose_flag=%d, DIM=%d nthreads=%d K=%d, input_file=%s, npoints=%d.\n", sort_flag, check_flag, verbose_flag, DIM, nthreads, K, input_file, npoints);
    	int numPoints = npoints;
	int startIndex = procRank * (numPoints/numProcs);
	int endIndex = (procRank == (numProcs - 1))?numPoints:((procRank+1) * (numPoints/numProcs));
	if(numPoints < numProcs)
	{
		if(procRank >= numPoints)
		{
			startIndex = 0;
			endIndex = 0;
		}
		else
		{
			startIndex = procRank;
			endIndex = procRank+1;
		}
	}
    SAFE_CALLOC(clusters, K, sizeof(ClusterPoint));
    SAFE_CALLOC(points, (endIndex-startIndex), sizeof(DataPoint));
    
    char junkc[250];
    if(strcmp(input_file, "random") != 0) {
        FILE * in = fopen(input_file, "r");
        if(in == NULL) {
            fprintf(stderr, "Could not open %s\n", input_file);
            exit(1);
        }
        
        int junk;
        float data;
        
        for(k=0,i = 0; i < endIndex; i++) {
	    memset(junkc,0,250);
            if(fscanf(in, "%s", &junkc) != 1) {
                fprintf(stderr, "Input file not large enough.\n");
                exit(1);
            }
	    if(i >= startIndex)
            {
		points[k].idx = i;
		points[k].label = new char[strlen(junkc)+1];
		strcpy(points[k].label, junkc);
		points[k].coord = new float[DIM];
	    }
            for(j = 0; j < DIM; j++) {
                if(fscanf(in, "%f,", &data) != 1) {
                    fprintf(stderr, "Input file not large enough.\n");
                    exit(1);
                }
	    	if(i >= startIndex)
                {
			points[k].coord[j] = data;
			points[k].clusterId = -1;
		}
            }
	    if(i < startIndex)
		continue;
	    k++;
        }
        fclose(in);
    } else {
        for(k=0,i = 0; i < endIndex; i++) {
	    memset(junkc,0,250);
	    sprintf(junkc,"%d",rand() % DIM);
	    if(i >= startIndex)
            {
		points[k].idx = i;
	    	points[k].coord = new float[DIM];
	    	points[k].label = new char[strlen(junkc)+1];
	    	strcpy(points[k].label,junkc);
            	points[k].clusterId = -1;
	    }
            for(j = 0; j < DIM; j++) {
		float tmp = (float)rand() / RAND_MAX;
		if(i >= startIndex)
	                points[k].coord[j] = 1.0 + tmp;
            }
	    if(i >= startIndex)
		k++;
        }
    }
    read_input_full(input_file);
}

Node * construct_tree(ClusterPoint *clusters, int start_idx, int end_idx, int depth, Node* parent) {
	int i;

	if (start_idx > end_idx)
		return NULL;
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
        //printf("%dth cluster has %d points, coord: ", i, clusters[i].num_of_points);
        for (int j = 0; j < DIM; j ++) {
            fprintf(fp,"%f, ", clusters[i].pt.coord[j]);
            //printf("%f, ", clusters[i].pt.coord[j]);
        }
        fprintf(fp,"\n");
        //printf("\n");
    }
}

void read_input_full(char* input_file) {
    int i, j,k;
    int c;
    
    SAFE_CALLOC(tmppoints, npoints, sizeof(DataPoint));
    
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
	    memset(junkc,0,250);
            if(fscanf(in, "%s", &junkc) != 1) {
                fprintf(stderr, "Input file not large enough.\n");
                exit(1);
            }
	    tmppoints[i].idx = i;
	    tmppoints[i].label = new char[strlen(junkc)+1];
	    strcpy(tmppoints[i].label, junkc);
	    tmppoints[i].coord = new float[DIM];
            for(j = 0; j < DIM; j++) {
                if(fscanf(in, "%f,", &data) != 1) {
                    fprintf(stderr, "Input file not large enough.\n");
                    exit(1);
                }
	    tmppoints[i].coord[j] = data;
	    tmppoints[i].clusterId = -1;
            }
        }
        fclose(in);
    } else {
	srandom(0);
        for(i = 0; i < npoints; i++) {
	    memset(junkc,0,250);
	    sprintf(junkc,"%d",rand() % DIM);
		tmppoints[i].idx = i;
	    	tmppoints[i].coord = new float[DIM];
	    	tmppoints[i].label = new char[strlen(junkc)+1];
	    	strcpy(tmppoints[i].label,junkc);
            	tmppoints[i].clusterId = -1;
            for(j = 0; j < DIM; j++) {
		float tmp = (float)rand() / RAND_MAX;
	                tmppoints[i].coord[j] = 1.0 + tmp;
            }
        }
    }
}
