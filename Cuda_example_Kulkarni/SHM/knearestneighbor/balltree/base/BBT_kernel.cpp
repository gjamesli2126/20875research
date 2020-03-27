/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include "BBT_kernel.hpp"


void k_nearest_neighbor_search(node* node, datapoint* point, int pos) {
#ifdef TRACK_TRAVERSALS
	point->num_nodes_traversed++;
#endif
#ifdef METRICS
	node->numpointsvisited++;
#endif
    float dist = getDistance(point, node->pivot);
    neighbor* pair_list = &nearest_neighbor[pos];
    if (dist > -0.000001 && pair_list->dist <= (dist - node->rad))
        return;
    else if (node->left == NULL && node->right == NULL) {
        if (dist > pair_list->dist)
            return;
        pair_list->dist = dist;
        pair_list->point = node->pivot;
    
        float f_temp = 0.0;
        datapoint* d_temp = NULL;
        for(int n = 1; n < K && (pair_list + n-1)->dist < (pair_list + n)->dist; n++) {
            f_temp = (pair_list + n)->dist;
            d_temp = (pair_list + n)->point;
            (pair_list + n)->dist = (pair_list + n-1)->dist;
            (pair_list + n)->point = (pair_list + n-1)->point;
            (pair_list + n-1)->dist = f_temp;
            (pair_list + n-1)->point = d_temp;
        }
    } else {
        float leftPivotDist = getDistance2(point, node->left->pivot);
        float rightPivotDist = getDistance2(point, node->right->pivot);
//        float leftBallDist = leftPivotDist - node->left->rad;
//        float rightBallDist = rightPivotDist - node->right->rad;
        
//        if (leftBallDist < 0 && rightBallDist < 0) {
            if (leftPivotDist < rightPivotDist) {
                k_nearest_neighbor_search(node->left, point, pos);
                k_nearest_neighbor_search(node->right, point, pos);
            } else {
                k_nearest_neighbor_search(node->right, point, pos);
                k_nearest_neighbor_search(node->left, point, pos);
            }
/*        } else {
            if (leftBallDist < rightBallDist) {
                k_nearest_neighbor_search(node->left, point, pos);
                k_nearest_neighbor_search(node->right, point, pos);
            } else {
                k_nearest_neighbor_search(node->right, point, pos);
                k_nearest_neighbor_search(node->left, point, pos);
            }
        }*/
    }
}

