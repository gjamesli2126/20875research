/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include "nn_data_types.h"

#ifdef TRACK_TRAVERSALS
int Point::next_id = 0;
int Node::next_id = 0;
#endif

Point::Point()
{
	coord = new float[DIM];
	for(int i = 0; i < DIM; i ++)
	{
		coord[i] = 0.0f;
	}
	closest_dist = FLT_MAX;
#ifdef TRACK_TRAVERSALS
	num_nodes_traversed = 0;
	id = next_id++;
#endif
}

Point::~Point() 
{
	delete [] coord;
}

Node::Node() 
{
	min = new float[DIM];
	max = new float[DIM];
	for (int i = 0; i < DIM; i++) 
	{
		min[i] = FLT_MAX;
		max[i] = 0;
	}
	for (int i = 0; i < MAX_POINTS_IN_CELL; i++) 
	{
		points[i] = NULL;
	}
	left = NULL;
	right = NULL;
#ifdef TRACK_TRAVERSALS
	id = next_id++;
#endif
}

Node::~Node() 
{
	delete [] min;
	delete [] max;
}

