/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include "kmeans_kernel.hpp"

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


void KdKmeansCPU(ClusterPoint* clusters, DataPoint* point, int procRank, int numProcs, mpi::communicator& world) {
    int i = -1, inCluster = -1;
    bool changed = false;
    int* tempClusterId = NULL;
    int* nearestClusters = NULL;
    Node* root = NULL;
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

    tempClusterId = (int*) malloc (sizeof(int) * (endIndex-startIndex));
    nearestClusters = (int*) malloc (sizeof(int) * (endIndex-startIndex));
    float* global_coord=new float[DIM];
    do {
        for(int i = 0; i < K; i ++) {
            clusters[i].num_of_points = 0;
        }
        for(int i = 0; i < (endIndex-startIndex); i ++) {
            tempClusterId[i] = points[i].clusterId;
        }

        for(int i = 0; i < K; i ++) {
            clusters[i].pt.clusterId = i;
        }

        root = construct_tree(clusters, 0, K - 1, 0, NULL);

        TIME_RESTART(KdKmeans_NN);
	for(int i=0;i<(endIndex-startIndex);i++)
	        NearestNeighbor(root, clusters, &points[i], &nearestClusters[i]);    
	for(int i=0;i<K;i++)
	{
	    for(int j=0; j<DIM;j++)
	    {
		clusters[i].pt.coord[j] = 0;
	    }
	}
       // For each point, find the nearest (cluster) centroid 
        for(int i = 0; i < (endIndex-startIndex); i ++) {
	    
            inCluster = nearestClusters[i];
            clusters[inCluster].num_of_points ++;
            points[i].clusterId = inCluster;
	    for(int j=0; j<DIM;j++)
	    {
		clusters[inCluster].pt.coord[j] += points[i].coord[j];
	    }
        }

        TIME_END(KdKmeans_NN);

	for(int i=0;i<K;i++)	
	{	
		int total_points;
		all_reduce(world,clusters[i].num_of_points,total_points,std::plus<int>());
		clusters[i].num_of_points = total_points;
		memset(global_coord, 0, sizeof(float)*DIM);
		all_reduce(world, clusters[i].pt.coord, DIM,global_coord, std::plus<float>());
		memcpy(clusters[i].pt.coord,global_coord,sizeof(float)*DIM);
		/*float global_coord;
		for(int d=0;d<DIM;d++)
		{
			all_reduce(world, clusters[i].pt.coord[d], global_coord, std::plus<float>());
			clusters[i].pt.coord[d] =global_coord;
		}*/
		
	}

	/*printf("end of iteration %d (newcoords of cluster)\n",total_iterations);
	char tmpstr[32];
	sprintf(tmpstr,"delme%d.txt",procRank);
	FILE* tmpfp = fopen(tmpstr,"w");
	PrintClusters(tmpfp);
	fclose(tmpfp);*/
        // Compute new centroid
        for (int i = 0; i < K; i ++) {
            assert(clusters[i].num_of_points != 0);
            for (int j = 0; j < DIM; j ++) {
                clusters[i].pt.coord[j] /= clusters[i].num_of_points;
            }
        }

        // check if anything has changed
        changed = false;
        for (int i = 0; i < (endIndex-startIndex); i ++) {
            if (points[i].clusterId != tempClusterId[i]) {
                changed = true;
                break;
            }
        }
	int change1=0, global_change1=0;
	if(changed)
		change1=1;
	all_reduce(world, change1, global_change1, std::plus<int>());
	if(global_change1 > 0)
		changed = true;
        avgnodes += nnodes;
        deconstruct_tree(root);
#ifdef TRACK_TRAVERSALS
	/*if(total_iterations == 0)
	{total_iterations++;
		break;}*/
	total_iterations++;
#endif
    } while (changed);

    	
    delete [] global_coord;	
    free(tempClusterId);
    free(nearestClusters);
#ifdef TRACK_TRAVERSALS
        for (int i = 0; i < (endIndex-startIndex); i ++)
    	    sum_nodes_traversed += points[i].num_nodes_traversed;
#endif
    //TIME_ELAPSED_PRINT(KdKmeans_NN, stdout);
    TIME_ELAPSED(KdKmeans_NN);
    float max_traversal_time = KdKmeans_NN_elapsed;
    reduce(world, KdKmeans_NN_elapsed, max_traversal_time, boost::parallel::maximum<float>(),0);	 
    if(procRank == 0)
	printf("Traversal time %f seconds\n",max_traversal_time/1000);
}

void NearestNeighbor(Node* root, ClusterPoint* clusters, DataPoint* point, int* clusterId) {
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
#ifdef TRACK_TRAVERSALS
    	(point->num_nodes_traversed)++;
#endif
            // Be careful about this line
            // If this cluster has alraady been checked, just ignore it
            DataPoint *target = &(curr->pivot->pt);
            if (!visited[target->clusterId]) {
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
    *clusterId = bestNode->pivot->pt.clusterId;
    //return bestNode->pivot;
}























