/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include "harness.h"
#ifndef BH_H_
#define BH_H_

#include <cuda.h>
//#define USE_LOCAL_STACK

enum {
	//Arg_binary,
	Arg_inputfile,
	Arg_npoints,
	Arg_num,
};

struct Vec {
public:
	float x;
	float y;
	float z;
};

const int MAX_POINTS_IN_CELL = 1;

struct Point {
public:
	float mass;
	Vec cofm; /* center of mass */
	Vec vel; /* current velocity */
	Vec acc; /* current acceleration */
	int id;
#ifdef TRACK_TRAVERSALS
	int num_nodes_traversed;
private:
	static int next_id;
#endif
} ;

struct Node {
public:
	float mass;
	Vec cofm;
	float ropen;
	Point* points[MAX_POINTS_IN_CELL];
	struct Node *left;
	struct Node *right;
	bool leafNode;
	int point_id;
#ifdef TRACK_TRAVERSALS
	int id;
private:
	static int next_id;
#endif
};


void print_treetofile(FILE* fp);
void print_preorder(Node* node, FILE*fp);
int app_main(int argc, char **argv);
void read_input(int argc, char **argv);
void compute_force(int start, int end);
void recursive_compute_force(Point *point, Node *root);
Node* construct_tree(Point *points, int start_idx, int end_idx, int depth, float mass, float rad, Vec& cofm);

#endif /* BH_H_ */
