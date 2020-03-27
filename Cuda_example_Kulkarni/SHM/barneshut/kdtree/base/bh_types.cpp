/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
int DIM=3;

#include "bh_types.h"
#include "common.h"

#ifdef TRACK_TRAVERSALS
int Point::next_id = 0;
int Node::next_id = 0;
#endif

Point::Point() {
#ifdef TRACK_TRAVERSALS
	num_nodes_traversed = 0;
	id = next_id++;
#endif
}

Point::~Point() {
}

Node::Node() {
	for (int i = 0; i < MAX_POINTS_IN_CELL; i++) {
		points[i] = NULL;
	}
	left = NULL;
	right = NULL;
	leafNode=false;
	mass=0.0;
	cofm.x=cofm.y=cofm.z=0.0;
	ropen=0.0;
#ifdef TRACK_TRAVERSALS
	id = next_id++;
#endif

#ifdef METRICS
	numPointsVisited=0;
#endif
}

Node::~Node() {
}

