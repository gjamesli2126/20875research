/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef PC_TYPES_H_
#define PC_TYPES_H_

enum {
	//Arg_binary,
	Arg_inputfile,
	Arg_npoints,
	Arg_num,
};

class Vec {
public:
	float x;
	float y;
	float z;

	Vec() {}
	Vec(float f) { x = f; y = f; z = f; }
};

const int MAX_POINTS_IN_CELL = 1;

class Point {
public:
	Point();
	~Point();
	float mass;
	Vec cofm; /* center of mass */
	Vec vel; /* current velocity */
	Vec acc; /* current acceleration */
	Point(const Point* p){mass=p->mass;cofm=p->cofm;vel=p->vel;acc=p->acc;id=p->id;}
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
	float mass;
	Vec cofm;
	float ropen;
	Point* points[MAX_POINTS_IN_CELL];
	Node *left;
	Node *right;
	bool leafNode;
#ifdef METRICS
	int numPointsVisited;
#endif

#ifdef TRACK_TRAVERSALS
	int id;
private:
	static int next_id;
#endif
};

#endif /* PC_TYPES_H_ */
