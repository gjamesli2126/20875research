/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
//
//  functions.cpp
//  BallTree-kNN

#include "knn_functions.h"

void read_input(int argc, char **argv) {
    int i, j;
    int c;
    char* input_file = NULL;
    
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
    
    printf("configuration: sort_flag=%d, check_flag = %d, verbose_flag=%d, K=%d, input_file=%s, npoints=%d, nsearchpoints=%d, DIM = %d, SPLICE_DEPTH = %d\n", sort_flag, check_flag, verbose_flag, K, input_file, npoints, nsearchpoints, DIM, SPLICE_DEPTH);
    
    SAFE_CALLOC(points, npoints, sizeof(node));
    SAFE_CALLOC(search_points, nsearchpoints, sizeof(node));
//    SAFE_MALLOC(nearest_neighbor, sizeof(neighbor)*nsearchpoints*K);
//    for (int i = 0; i < nsearchpoints*K; i ++) {
//        nearest_neighbor[i].dist = FLT_MAX;
//        nearest_neighbor[i].point = NULL;
//    }
	SAFE_MALLOC(nearest_distance, sizeof(float)*nsearchpoints*K);
	SAFE_MALLOC(nearest_point_index, sizeof(unsigned int)*nsearchpoints*K);
    
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
            if(fscanf(in, "%d", &junk) != 1) {
                fprintf(stderr, "Input file not large enough.\n");
                exit(1);
            }
            points[i].label = junk;
            for(j = 0; j < DIM; j++) {
                if(fscanf(in, "%f", &data) != 1) {
                    fprintf(stderr, "Input file not large enough.\n");
                    exit(1);
                }
                points[i].coord[j] = data;
            }
        }
        
        for(i = 0; i < nsearchpoints; i++) {
            search_points[i].idx = i;
            if(fscanf(in, "%d", &junk) != 1) {
                fprintf(stderr, "Input file not large enough.\n");
                exit(1);
            }
            search_points[i].label = junk;
            for(j = 0; j < DIM; j++) {
                if(fscanf(in, "%f", &data) != 1) {
                    fprintf(stderr, "Input file not large enough.\n");
                    exit(1);
                }
                search_points[i].coord[j] = data;
                //search_points[i*DIM + j] = data;
            }
        }
        fclose(in);
        
    } else {
        for(i = 0; i < npoints; i++) {
            points[i].idx = i;
            points[i].label = rand() % DIM;
            for(j = 0; j < DIM; j++) {
                points[i].coord[j] = 1.0 + (float)rand() / RAND_MAX;
            }
        }
        
        for(i = 0; i < nsearchpoints; i++) {
            search_points[i].idx = i;
            search_points[i].label = rand() % DIM;
            for(j = 0; j < DIM; j++) {
                //search_points[i*DIM + j] = 1.0 + (float)rand() / RAND_MAX;			
                search_points[i].coord[j] = 1.0 + (float)rand() / RAND_MAX;
            }
        }
    }
}


node * construct_tree(datapoint *points, int start, int end, datapoint** &datalist, int depth, int id){
    node* root = new node();
    int total = end - start + 1;
    nnodes ++;
    root->pre_id = id;
    root->depth = depth;
    if (depth > max_depth)
        max_depth = depth;
    if (total <= 1) {
        root->pivot = datalist[start];
        return root;
    }

    root->pivot = new datapoint();
    for (int i = 0; i < DIM; i ++) {
        root->pivot->coord[i] = 0.0;
        for (int j = start; j <= end; j ++) {
            root->pivot->coord[i] += datalist[j]->coord[i];
        }
        root->pivot->coord[i] /= total;
    }
    pair<float, datapoint*> rad_furthest = getRadius(root->pivot, start, end, datalist);
    root->rad = rad_furthest.first;
    datapoint* furthest1 = rad_furthest.second;
//    float* DistList = NULL;
//    SAFE_CALLOC(DistList, total, sizeof(float));
    vector<float> DistList;
    DistList.resize(total);
    datapoint* furthest2 = getMaxDist(furthest1, start, end, datalist, DistList);
    int numRight = 0;
    for (int i = start; i <= end - numRight; i ++) {
        int j = i - start;
        float dist = getDistance2(furthest2, datalist[i]);
        if (dist < DistList[j]) {
            datapoint* tmp = datalist[i];
            int pos = end - numRight;
            datalist[i] = datalist[pos];
            datalist[pos] = tmp;
            
            pos -= start;
            float val = DistList[j];
            DistList[j] = DistList[pos];
            DistList[pos] = val;
            numRight ++;
            i --;
        }
    }
//    free(DistList);
//    DistList = NULL;
    DistList.clear();
    root->left = construct_tree(points, start, end - numRight, datalist, depth + 1, 2 * id);
    root->right = construct_tree(points, end - numRight + 1, end, datalist, depth + 1, 2 * id + 1);
    
    return root;
}

float getDistance(datapoint* key, datapoint* curr) {
    float dist = 0.0;
    for (int i = 0; i < DIM; ++i) {
        dist += (key->coord[i] - curr->coord[i]) * (key->coord[i] - curr->coord[i]);
    }
    return sqrt(dist);
}

float getDistance2(datapoint* key, datapoint* curr) {
    float dist = 0.0;
    for (int i = 0; i < DIM; ++i) {
        dist += (key->coord[i] - curr->coord[i]) * (key->coord[i] - curr->coord[i]);
    }
    return dist;
}

pair<float, datapoint*> getRadius(datapoint* target, int start, int end, datapoint** &datalist) {
    float rad = 0.0;
    datapoint *furthest = NULL;
    for (int i = start; i <= end; i++) {
        float dist = getDistance2(target, datalist[i]);
        if(dist > rad){
            rad = dist;
            furthest = datalist[i];
        }
    }
    return make_pair(sqrt(rad), furthest);
}

datapoint* getMaxDist(datapoint* target, int start, int end, datapoint** &datalist, vector<float> &distlist) {
    float max_dist = 0.0;
    datapoint *furthest = NULL;
    for (int i = start; i <= end; i++) {
        int j = i - start;
        distlist[j] = getDistance2(target, datalist[i]);
        if(distlist[j] > max_dist){
            max_dist = distlist[j];
            furthest = datalist[i];
        }
    }
    return furthest;
}

void sort_search_points(datapoint* points, int start, int size) {
    if (size < 5)
        return;
    
    float minVal[DIM];
    float maxVal[DIM];
    float diff[DIM];
    float maxDiff = - 1.0;
    
    for (int i = 0; i < DIM; i ++) {
        minVal[i] = FLT_MAX;
        maxVal[i] = FLT_MIN;
    }
    for (int i = start; i < start + size; i ++) {
        for (int j = 0; j < DIM; j ++) {
            minVal[j] = std::min(minVal[j], points[i].coord[j]);
            maxVal[j] = std::max(maxVal[j], points[i].coord[j]);
        }
    }
    
    for (int i = 0; i < DIM; i ++) {
        diff[i] = maxVal[i] - minVal[i];
    }
    for (int i = 0; i < DIM; i ++) {
        if (diff[i] > maxDiff) {
            maxDiff = diff[i];
            label = i;
        }
    }
    qsort(&points[start], size, sizeof(datapoint), cmpfn_float);
    int mid = size / 2;
    sort_search_points(points, start, mid);
    sort_search_points(points, start + mid, size - mid);
}

static int cmpfn_float(const void *a, const void *b) {
    datapoint* pa = (datapoint*) a;
    datapoint* pb = (datapoint*) b;
    
    if (pa->coord[label] < pb->coord[label])
        return -1;
    else if (pa->coord[label] == pb->coord[label])
        return 0;
    else
        return 1;
}

void print_result() {
    int correct = 0;
    for (int i = 0; i < nsearchpoints; i ++) {
        
        int cnts[MAX_LABEL];
        memset(&cnts, 0, sizeof(int) * MAX_LABEL);
        int max_label = 0;
        for(int j = K-1; j >= 0; j --)
            cnts[points[nearest_point_index[i * K + j]].label] ++;
        if (verbose_flag) {
        	printf("%d: ", i);
        	for(int j = K-1; j >= 0; j --) {
            	printf("%d (%1.3f), ", nearest_point_index[i * K + j], nearest_distance[i * K + j]);
        	}
        	printf("\n");
        }
        for (int j = 0; j < MAX_LABEL; j ++) {
            if (cnts[j] > cnts[max_label]) {
                max_label = j;
            }
        }
        if (search_points[i].label == max_label) {
            correct ++;
        }
    }
    printf("The ratio is %f.\n", (float) correct / nsearchpoints);
}

void printTree(node* root, int tabs) {
    if (root == NULL)
        return;
    for (int i = 0 ; i < tabs; i ++)
        printf(" ");
    printf("%d node: rad=%f, (%f, %f, %f)\n", root->pre_id, root->rad, root->pivot->coord[0], root->pivot->coord[1], root->pivot->coord[2]);
    
    printTree(root->left, tabs + 1);
    printTree(root->right, tabs + 1);
}

