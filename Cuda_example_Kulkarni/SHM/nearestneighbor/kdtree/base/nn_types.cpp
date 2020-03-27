/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
int DIM;

#include "nn_types.h"
#include "common.h"

#ifdef TRACK_TRAVERSALS
int Point::next_id = 0;
int Node::next_id = 0;
#endif

Point::Point() {
	coord = new float[DIM];
	closest_dist = FLT_MAX;
#ifdef TRACK_TRAVERSALS
	num_nodes_traversed = 0;
	id = next_id++;
#endif
}

Point::~Point() {
	delete [] coord;
}

Node::Node() {
	min = new float[DIM];
	max = new float[DIM];
	for (int i = 0; i < DIM; i++) {
		min[i] = FLT_MAX;
		max[i] = -FLT_MAX;
	}
	for (int i = 0; i < MAX_POINTS_IN_CELL; i++) {
		points[i] = NULL;
	}
	left = NULL;
	right = NULL;
#ifdef TRACK_TRAVERSALS
	id = next_id++;
#endif
#ifdef METRICS
	numPointsVisited=0;
#endif

}

Node::~Node() {
	delete [] min;
	delete [] max;
}

