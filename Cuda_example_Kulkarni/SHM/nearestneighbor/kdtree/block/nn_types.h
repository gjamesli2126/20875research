/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef NN_TYPES_H_
#define NN_TYPES_H_
//#include<vector>
//using namespace std;

//#define TRACK_TRAVERSALS
const int MAX_POINTS_IN_CELL = 1;

class Point {
public:
	Point();
	~Point();
	//std::vector<int> nodesVisited;
	int label;
	float *coord;
	int closest_label;
	float closest_dist;
#ifdef TRACK_TRAVERSALS
	int num_nodes_traversed;
	int id;
private:
	static int next_id;
#endif
};

class Node {
public:
	Node();
	~Node();

	//int level;
	int axis;
	float splitval;
	float *min;
	float *max;
	Point* points[MAX_POINTS_IN_CELL];
	Node *left;
	Node *right;
#ifdef TRACK_TRAVERSALS
	int id;
private:
	static int next_id;
#endif
};

#endif /* NN_TYPES_H_ */
