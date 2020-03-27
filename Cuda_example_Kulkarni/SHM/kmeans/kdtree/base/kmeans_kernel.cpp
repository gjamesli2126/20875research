/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include "kmeans_kernel.hpp"

#ifdef PAPI
double missRate=0., CPI=0.;
long long totIns=0;
int numSamples=0;
#endif

TIME_INIT(Kmeans_NN);
TIME_INIT(KdKmeans_NN);

float GetDistance(DataPoint* pt, DataPoint* cls) {
    float dist = 0.0f;
    for (int i = 0; i < DIM; i ++) {
        dist += (pt->coord[i] - cls->coord[i]) * (pt->coord[i] - cls->coord[i]);
    }
    return sqrt(dist);
}

void KmeansCPU () {
    bool changed = false;
    DataPoint* temp_points = new DataPoint [K];
	int* old_cluster_idx = new int [npoints];
	int nearestCluster;
    float min_dist = 0.0;    
    int times = 0;
    
    do {
        times ++;
        memset(temp_points, 0, sizeof(DataPoint) * K);
		memset(old_cluster_idx, 0, sizeof(int) * npoints);
        for (int i = 0; i < K; i ++) {
            clusters[i].num_of_points = 0;
        }
        
        for (int  i = 0; i < npoints; i ++) {
            old_cluster_idx[i] = points[i].clusterId;
        }		

        for (int i = 0; i < npoints; i ++) {
            TIME_RESTART(Kmeans_NN);
			min_dist = GetDistance(&points[i], &clusters[0].pt);
            nearestCluster = 0;
			for (int j = 1; j < K; j ++) {
				double dist = GetDistance(&points[i], &clusters[j].pt);
				if (dist < min_dist) {
					nearestCluster = j;
					min_dist = dist;
				}
			}
            TIME_END(Kmeans_NN);
			clusters[nearestCluster].num_of_points ++;
			points[i].clusterId = nearestCluster;
			for (int j = 0; j < DIM; j ++) {
				temp_points[nearestCluster].coord[j] += points[i].coord[j];
			}
		}
        for (int i = 0; i < K; i ++) {
            assert (clusters[i].num_of_points != 0);
            for (int j = 0; j < DIM; j ++) {
                clusters[i].pt.coord[j] = temp_points[i].coord[j] / clusters[i].num_of_points;
            }
        }
        
        changed = false;
        for (int i = 0; i < npoints; i ++) {
            if (points[i].clusterId != old_cluster_idx[i]) {
                changed = true;
                break;
            }
        }
    } while (changed);
    printf("Total run times: %d\n", times);
    TIME_ELAPSED_PRINT(Kmeans_NN, stdout);
}

void* thread_function(void *arg) {
    struct thread_args *args = (struct thread_args*)arg;
    int j = 0;
    for(j = args->lb; j < args->ub; j++) {
        args->nearestclusters[j] = NearestNeighbor(args->root, clusters, &points[j]);           
    }
}

void KdKmeansCPU(ClusterPoint* clusters, DataPoint* point) {
    int i = -1, inCluster = -1;
    bool changed = false;
    Coord* tempCoords = NULL;
    int* tempClustedId = NULL;
    ClusterPoint** nearestClusters = NULL;
    Node* root = NULL;

    tempCoords = (Coord*) malloc (sizeof(Coord) * K);
    for(int i=0;i<K;i++)
    {	
	tempCoords[i].coord = new float[DIM];
	memset(tempCoords[i].coord,0,sizeof(float)*DIM);
    }

    tempClustedId = (int*) malloc (sizeof(int) * npoints);
    nearestClusters = (ClusterPoint**) malloc (sizeof(ClusterPoint*) * npoints);
#ifdef PAPI
    int retval = PAPI_start(eventSet);
    if (retval != PAPI_OK) handle_error(retval);
#endif

    do {
	for(int i=0;i<K;i++)
    	{	
		memset(tempCoords[i].coord,0,sizeof(float)*DIM);
    	}
        for(int i = 0; i < K; i ++) {
            clusters[i].num_of_points = 0;
        }
        for(int i = 0; i < npoints; i ++) {
            tempClustedId[i] = points[i].clusterId;
        }

        for(int i = 0; i < K; i ++) {
            clusters[i].pt.clusterId = i;
        }
        root = construct_tree(clusters, 0, K - 1, 0, NULL);


        struct thread_args *args;
        pthread_t *threads;
        SAFE_MALLOC(args, sizeof(struct thread_args)*nthreads);
        SAFE_MALLOC(threads, sizeof(pthread_t)*nthreads);
        int start = 0;
        for(int j = 0; j < nthreads; j++) {
            int num = (npoints - start) / (nthreads - j);
            args[j].tid = j;
            args[j].lb = start;
            args[j].ub = start + num;
            args[j].root = root;
            args[j].nearestclusters = nearestClusters;
            start += num;
            //printf("%d %d\n", args[j].lb, args[j].ub);
        }
    
        TIME_RESTART(KdKmeans_NN);
        for(int j = 0; j < nthreads; j++) {
            if(pthread_create(&threads[j], NULL, thread_function, &args[j]) != 0) {
                fprintf(stderr, "Could not create thread %d\n", j);
                exit(1);
            }
        }

        for(int j = 0; j < nthreads; j++) {
            pthread_join(threads[j], NULL);
        }       
	
	#ifdef METRICS
	printLoadDistribution(true);
	subtrees.clear();
	#endif 
        // For each point, find the nearest (cluster) centroid 
        for(int i = 0; i < npoints; i ++) {
            // continue coding;
//            nearestCluster = NearestNeighbor(root, clusters, &points[i]);
            inCluster = nearestClusters[i]->pt.clusterId;
            clusters[inCluster].num_of_points ++;
            for(int j = 0; j < DIM; j ++) {
                tempCoords[inCluster].coord[j] += points[i].coord[j];
            }
            points[i].clusterId = inCluster;
        }

        // Compute new centroid
        for (int i = 0; i < K; i ++) {
            assert(clusters[i].num_of_points != 0);
            for (int j = 0; j < DIM; j ++) {
                clusters[i].pt.coord[j] = tempCoords[i].coord[j] / clusters[i].num_of_points;
            }
        }

        // check if anything has changed
        changed = false;
        for (int i = 0; i < npoints; i ++) {
            if (points[i].clusterId != tempClustedId[i]) {
                changed = true;
                break;
            }
        }
        TIME_END(KdKmeans_NN);
        free(args);
        free(threads);
	avgnodes += nnodes;
        deconstruct_tree(root);
	total_iterations++;
    } while (changed);

#ifdef PAPI
    /* Stop the counters */
    retval = PAPI_stop(eventSet, values);
    if (retval != PAPI_OK) handle_error(retval);
#endif

    for(int i=0;i<K;i++)
	delete [] tempCoords[i].coord;
    free(tempCoords);
    free(tempClustedId);
    free(nearestClusters);
#ifdef TRACK_TRAVERSALS
        for (int i = 0; i < npoints; i ++)
    	    sum_nodes_traversed += points[i].num_nodes_traversed;
#endif
    //TIME_ELAPSED_PRINT(KdKmeans_NN, stdout);
    TIME_ELAPSED(KdKmeans_NN);
    printf("Traversal time %f seconds\n",KdKmeans_NN_elapsed/1000);
#ifdef PAPI
	double tmp = values[0]/(double)(values[1]);
	double tmp2 = values[2]/(double)(values[3]);
    //printf("Avg L3 Miss Rate / step:%f Avg CPI / step :%f Avg Total number of instructions / step :%ld\n",missRate/numSamples,CPI/numSamples, totIns);
    printf("Avg L3 Miss Rate / step:%f Avg CPI / step :%f Avg Total number of instructions / step :%ld\n",tmp,tmp2, totIns);
#endif
}

ClusterPoint* NearestNeighbor(Node* root, ClusterPoint* clusters, DataPoint* point) {
    Node* curr = root;
    Node* prev = NULL;
    int axis = -1;
    double bestDist = FLT_MAX;
    Node* bestNode = NULL;
    double dist = FLT_MAX;
    Node* subRoot = root;

    bool* visited = (bool*) malloc (sizeof(bool) * K);
    memset(visited, 0, sizeof(bool)*K);

    while(curr != NULL) {
        prev = NULL;
        curr = subRoot;

        while(curr != NULL) {
            axis = curr->axis;
            prev = curr;
            if (point->coord[axis] <= curr->pivot->pt.coord[axis]) {
                curr = curr->left;
            } else {
                curr = curr->right;
            }
        }

        curr = prev;
        prev = NULL;
        while (curr != NULL) {
            // Be careful about this line
            // If this cluster has alraady been checked, just ignore it
            DataPoint *target = &(curr->pivot->pt);
            if (!visited[target->clusterId]) {
#ifdef TRACK_TRAVERSALS
    		(point->num_nodes_traversed)++;
#endif
#ifdef METRICS
		curr->numpointsvisited++;
#endif
                visited[target->clusterId] = true;
                dist = GetDistance(point, target);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestNode = curr;
                }

                // See whether need to visit sibling subtree node
                axis = curr->axis;
                dist = MOD(point->coord[axis] - target->coord[axis]);
                if (dist < bestDist) {
                    if (curr->left == prev && curr->right != NULL) {
                        subRoot = curr->right;
                        break;
                    } else if (curr->left != NULL) {
                        subRoot = curr->left;
                        break;
                    }
                }
            }
            prev = curr;
            curr = curr->parent;
        }
    }
    free(visited);
    assert(bestNode != NULL && bestDist < FLT_MAX);
    return bestNode->pivot;
}























