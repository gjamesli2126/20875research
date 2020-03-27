/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <float.h>
#include <iostream>
using namespace std;

#include "harness.h"

class Vec {
public:
	float x;
	float y;
	float z;

	Vec() {}
	Vec(float f) { x = f; y = f; z = f; }
};

enum bh_node_type {
	bhLeafNode,
	bhNonLeafNode
};

class Point {
public:
	Point() {
#ifdef TRACK_TRAVERSALS
	id = next_id++;
#endif
	}
	float mass;
	Vec cofm; /* center of mass */
	Vec vel; /* current velocity */
	Vec acc; /* current acceleration */
#ifdef TRACK_TRAVERSALS
	int num_nodes_traversed;
	int id;
private:
	static int next_id;
#endif
};

class Node {
public:
	Node(Point *p) : point(p), type(bhLeafNode) 
	{ 
		clearChildren(); 
		#ifdef METRICS 
		depth=0;
		numpointsvisited=0;
		#endif
	}
	Node() : point(new Point), type(bhNonLeafNode) 
	{ 
		clearChildren(); 
		#ifdef METRICS 
		depth=0;
		numpointsvisited=0;
		#endif
	}
	~Node() {
		if (type == bhNonLeafNode) delete point;
	}

	Node *getChild(int i);
	void setChild(int i, Node *c);

	enum bh_node_type type;
	Point *point;
	Node *child0;
	Node *child1;
	Node *child2;
	Node *child3;
	Node *child4;
	Node *child5;
	Node *child6;
	Node *child7;
#ifdef METRICS
	int depth;
	int numpointsvisited;
#endif

private:
	void clearChildren();
};

#ifdef TRACK_TRAVERSALS
int Point::next_id = 0;
#endif

#ifdef METRICS
class subtreeStats
{
public:
	long int footprint;
	int numnodes;
	subtreeStats(){footprint=0;numnodes=0;}
};
int splice_depth;
int max_depth=0;
std::vector<Node*> subtrees;
void getSubtreeStats(Node* ver, subtreeStats* stat, int DOR);
void printLoadDistribution();
#endif

void Node::clearChildren() {
	child0 = NULL;
	child1 = NULL;
	child2 = NULL;
	child3 = NULL;
	child4 = NULL;
	child5 = NULL;
	child6 = NULL;
	child7 = NULL;
}

Node *Node::getChild(int i) {
	if (i == 0) return child0;
	else if (i == 1) return child1;
	else if (i == 2) return child2;
	else if (i == 3) return child3;
	else if (i == 4) return child4;
	else if (i == 5) return child5;
	else if (i == 6) return child6;
	else return child7;
}

void Node::setChild(int i, Node *c) {
	if (i == 0) child0 = c;
	else if (i == 1) child1 = c;
	else if (i == 2) child2 = c;
	else if (i == 3) child3 = c;
	else if (i == 4) child4 = c;
	else if (i == 5) child5 = c;
	else if (i == 6) child6 = c;
	else child7 = c;
}

#pragma afterClassDecs

uint64_t nbodies;
uint64_t ntimesteps;
float dtime;
float eps;
float tol;
float half_dtime;
float inv_tol_squared;
float eps_squared;
float diameter;
uint64_t sortidx;
Node *g_root;
Point **points_sorted;
int global_timestep;

enum {
	//Arg_binary,
	Arg_inputfile,
	Arg_npoints,
	Arg_num,
};

Point* read_input(int argc, char **argv) {

	if ((argc!=Arg_num)) {
		fprintf(stderr, "Usage: [input_file] [nbodies]\n");
		exit(1);
	}

	FILE *infile=fopen(argv[Arg_inputfile], "r");
	if (( ! infile)) {
		fprintf(stderr, "Error: could not read input file: %s\n", argv[Arg_inputfile]);
		exit(1);
	}

	nbodies=0;
	nbodies=atoll(argv[Arg_npoints]);
	if ((nbodies<=0))
	{
		fprintf(stderr, "Error: nbodies not valid.\n");
		exit(1);
	}

	printf("Overriding nbodies from input file. nbodies = %lld\n", nbodies);
	uint64_t junk;
	fscanf(infile, "%lld", ( & junk));

	if ((nbodies<=0))
	{
		fscanf(infile, "%lld", ( & nbodies));
		if ((nbodies<1))
		{
			fprintf(stderr, "Error: nbodies must be at least 1!\n");
			exit(1);
		}
	}

	fscanf(infile, "%lld", ( & ntimesteps));
	if ((ntimesteps<1))
	{
		fprintf(stderr, "Error: ntimesteps must be at least 1!\n");
		exit(1);
	}

	fscanf(infile, "%f", ( & dtime));
	if ((dtime<=0.0))
	{
		fprintf(stderr, "Error: dtime can not be zero!\n");
		exit(1);
	}

	fscanf(infile, "%f", ( & eps));
	fscanf(infile, "%f", ( & tol));
	half_dtime=(0.5*dtime);
	inv_tol_squared=(1.0/(tol*tol));
	eps_squared=(eps*eps);
	Point *points = new Point[nbodies];

	for (int i=0; i<nbodies; i ++ ) {
		int ret = fscanf(infile, "%f %f %f %f %f %f %f", ( & points[i].mass), ( & points[i].cofm.x), ( & points[i].cofm.y), ( & points[i].cofm.z), ( & points[i].vel.x), ( & points[i].vel.y), ( & points[i].vel.z));
		if (ret!=7) {
			fprintf(stderr, "Error: Invalid point (%d).\n", i);
			exit(1);
		}
		points[i].acc.x=(points[i].acc.y=(points[i].acc.z=0.0));
		points[i].id=i;
#ifdef TRACK_TRAVERSALS
		points[i].num_nodes_traversed=0;
#endif
	}
	if ((infile!=stdin)) {
		fclose(infile);
	}

	return points;
}

void  compute_bounding_box(Point *points, Vec *center) {
	Vec pos;
	Vec min(FLT_MAX);
	Vec max(FLT_MIN);

	/* compute the max and min positions to form a bounding box */
	for (int i=0; i<nbodies; i ++ ) {
		pos=points[i].cofm;
		if ((min.x>pos.x))
		{
			min.x=pos.x;
		}
		if ((min.y>pos.y))
		{
			min.y=pos.y;
		}
		if ((min.z>pos.z))
		{
			min.z=pos.z;
		}
		if ((max.x<pos.x))
		{
			max.x=pos.x;
		}
		if ((max.y<pos.y))
		{
			max.y=pos.y;
		}
		if ((max.z<pos.z))
		{
			max.z=pos.z;
		}
	}

	/* compute the maximum of the diameters of all axes */
	diameter=(max.x-min.x);
	if ((diameter<(max.y-min.y)))
	{
		diameter=(max.y-min.y);
	}
	if ((diameter<(max.z-min.z)))
	{
		diameter=(max.z-min.z);
	}

	/* compute the center point */
	center->x=((max.x+min.x)*0.5);
	center->y=((max.y+min.y)*0.5);
	center->z=((max.z+min.z)*0.5);
}

void  insert_point(Node *root, Point *p, float r, int depth) {
	Vec offset(0.0f);
	assert(root != NULL);
	assert(p != NULL);
#ifdef METRICS
	if(depth > max_depth)
		max_depth=depth;
#endif
	/*
    From the root locate where this point will be:
	 * 		- 0 is lower-front-left
	 * 		- 1 is lower-front-right
	 * 		- 2 is lower-back-left
	 * 		- 3 is lower-back-right
	 * 		- 4 is upper-front-left
	 * 		- 5 is upper-front-right
	 * 		- 6 is upper-back-left
	 * 		- 7 is upper-back-right
	 *
	 * 1. Starting from the lower, front left compare the x component:
	 * 		- if p.x is greater than root.x then go right (add 1)
	 * 		- else stay left
	 *
	 * 		- if p.y is greater than root.y then go back (add 2)
	 * 		- else stay front
	 *
	 * 		- if p.z is greater than root.z then go up (add 4)
	 * 		- else stay low
	 */

	int space=0;
	Point *rootPoint = root->point;
	if ((rootPoint->cofm.x<p->cofm.x)) {
		space += 1;
		offset.x = r;
	}
	if ((rootPoint->cofm.y<p->cofm.y)) {
		space += 2;
		offset.y = r;
	}
	if ((rootPoint->cofm.z<p->cofm.z)) {
		space += 4;
		offset.z = r;
	}
	Node *child = root->getChild(space);
	if (child == NULL) {
		Node *n = new Node(p);
#ifdef METRICS
		n->depth=depth+1;
#endif
		root->setChild(space, n);
	} else {
		float half_r = 0.5f * r;
		if (child->type == bhLeafNode) {
			Node *newNode = new Node;
#ifdef METRICS
			newNode->depth=depth+1;
#endif
			Point *nodePoint = newNode->point;
			nodePoint->cofm.x=((rootPoint->cofm.x-half_r)+offset.x);
			nodePoint->cofm.y=((rootPoint->cofm.y-half_r)+offset.y);
			nodePoint->cofm.z=((rootPoint->cofm.z-half_r)+offset.z);
			insert_point(newNode, p, half_r, (depth+1));
			insert_point(newNode, child->point, half_r, (depth+1));
			root->setChild(space, newNode);
		} else {
			insert_point(child, p, half_r, (depth+1));
		}
	}
}

void free_tree(Node *root) {
	assert(root != NULL);
	for (int i = 0; i < 8; i++) {
		Node *child = root->getChild(i);
		if (child != NULL && child->type==bhNonLeafNode) {
			free_tree(child);
			root->setChild(i, NULL);
		}
	}
	delete root;
}

void  compute_center_of_mass(Node *node) {
	assert(node != NULL);
	int j = 0;
	Vec cofm(0.0f);
	float mass=0.0;
#if METRICS
			if(node->depth == splice_depth)
			{
				subtrees.push_back(node);
			}
#endif
	for (int i = 0; i < 8; i++) {
		Node *child = node->getChild(i);
		if (child != NULL) {
			/* compact child nodes for speed */
			if (i != j) {
				node->setChild(j, node->getChild(i));
				node->setChild(i, NULL);
			}
			j++;
			/* If non leave node need to traverse children: */
			if ((child->type==bhNonLeafNode)) {
				/* summarize children */
				compute_center_of_mass(child);
			} else {

#if METRICS
			if(child->depth == splice_depth)
			{
				subtrees.push_back(child);
			}
#endif
				/* insert this point in sorted order */
				points_sorted[(sortidx ++ )]=child->point;
			}
			Point *childPoint = child->point;
			mass+=childPoint->mass;
			cofm.x+=(childPoint->cofm.x*childPoint->mass);
			cofm.y+=(childPoint->cofm.y*childPoint->mass);
			cofm.z+=(childPoint->cofm.z*childPoint->mass);
		}
	}
	Point *nodePoint = node->point;
	nodePoint->cofm.x=(cofm.x/mass);
	nodePoint->cofm.y=(cofm.y/mass);
	nodePoint->cofm.z=(cofm.z/mass);
	nodePoint->mass=mass;
}

inline void update_point(Point *p, Node *root, const Vec& dr, float drsq, float epssq) {
	drsq+=epssq;
	float idr=(1.0/sqrt(drsq));
	float nphi=(root->point->mass*idr);
	float scale=((nphi*idr)*idr);
	p->acc.x+=(dr.x*scale);
	p->acc.y+=(dr.y*scale);
	p->acc.z+=(dr.z*scale);
}

void compute_force_recursive(Point *p, Node *node, float dsq, float epssq) {

	if (node == NULL) return;

#ifdef TRACK_TRAVERSALS	
	p->num_nodes_traversed++;
#endif
#ifdef METRICS
	node->numpointsvisited++;
#endif
	Vec dr;
	Point *nodePoint = node->point;
	dr.x=(nodePoint->cofm.x-p->cofm.x);
	dr.y=(nodePoint->cofm.y-p->cofm.y);
	dr.z=(nodePoint->cofm.z-p->cofm.z);
	float drsq=(((dr.x*dr.x)+(dr.y*dr.y))+(dr.z*dr.z));
	/*
    always do recursive to get most exact
    particle is far away can stop here
	 */
	if ((drsq<dsq)) {
		if ((node->type==bhNonLeafNode)) {
			dsq*=0.25;
			compute_force_recursive(p, node->child0, dsq, epssq);
			compute_force_recursive(p, node->child1, dsq, epssq);
			compute_force_recursive(p, node->child2, dsq, epssq);
			compute_force_recursive(p, node->child3, dsq, epssq);
			compute_force_recursive(p, node->child4, dsq, epssq);
			compute_force_recursive(p, node->child5, dsq, epssq);
			compute_force_recursive(p, node->child6, dsq, epssq);
			compute_force_recursive(p, node->child7, dsq, epssq);
		}
		else {
			/* two different particles */
			if ((p!=node->point)) {
				update_point(p, node, dr, drsq, epssq);
			}
		}
	}
	else {
		//drsq+=epssq;
		update_point(p, node, dr, drsq, epssq);
	}
}

void  advance_point(Point * p, float dthf, float dtime) {
	Vec delta_v;
	Vec velh;

	assert(p != NULL);

	delta_v.x=(p->acc.x*dthf);
	delta_v.y=(p->acc.y*dthf);
	delta_v.z=(p->acc.z*dthf);
	velh.x=(p->vel.x+delta_v.x);
	velh.y=(p->vel.y+delta_v.y);
	velh.z=(p->vel.z+delta_v.z);
	p->cofm.x+=(velh.x*dtime);
	p->cofm.y+=(velh.y*dtime);
	p->cofm.z+=(velh.z*dtime);
	p->vel.x=(velh.x+delta_v.x);
	p->vel.y=(velh.y+delta_v.y);
	p->vel.z=(velh.z+delta_v.z);
}

void compute_force(int start, int end, Point **points, Node *root, float size, float itolsq, int step, float dthf, float epssq) {
	float dsq = size * size * itolsq;
#pragma parallelForOnTree
	for(int i = start; i < end; i++) {
		Point *p = points[i];
		Vec a_prev;
		a_prev = p->acc;
		p->acc.x = p->acc.y = p->acc.z = 0.0;

		compute_force_recursive(p, root, dsq, epssq);

		if (step > 0) {
			Vec delta_v;
			delta_v.x = (p->acc.x - a_prev.x) * dthf;
			delta_v.y = (p->acc.y - a_prev.y) * dthf;
			delta_v.z = (p->acc.z - a_prev.z) * dthf;

			p->vel.x += delta_v.x;
			p->vel.y += delta_v.y;
			p->vel.z += delta_v.z;
		}
	}
}

void call_compute_force(int start, int end) {
	compute_force(start, end, points_sorted, g_root, diameter, inv_tol_squared, global_timestep, half_dtime, eps_squared);
}

int  app_main(int argc, char **argv) {

	Point *points = read_input(argc, argv);
	printf("Configuration: nbodies = %lld, ntimesteps = %lld, dtime = %f eps = %f, tol = %f\n", nbodies, ntimesteps, dtime, eps, tol);

	for (int t=0; t<ntimesteps; t ++ ) {
		printf("Time step %d:\n", t);

		double starttime=clock();
		Vec center;
		compute_bounding_box(points, &center);
		g_root = new Node;
		g_root->point->cofm = center;

		for (int i=0; i<nbodies; i ++ ) {
			insert_point(g_root, ( & points[i]), (diameter*0.5), 0);
		}
#ifdef METRICS
		splice_depth = 5;//max_depth/2;
		printf("max_depth %d splice_depth %d\n",max_depth, splice_depth);
#endif
		points_sorted = new Point*[nbodies];
		sortidx=0;
		compute_center_of_mass(g_root);

		if (!Harness::get_sort_flag()) { // unsort points
			for (int i = 0; i < nbodies; i++) {
				points_sorted[i] = &points[i];
			}
		}
		double endtime=clock();
		printf("Tree construction time: %f\n",(endtime-starttime)/(float)(CLOCKS_PER_SEC));
		Harness::start_timing();
		global_timestep=t;
//		compute_force(0, nbodies, points_sorted, g_root, diameter, inv_tol_squared,
//				global_timestep, half_dtime, eps_squared);
//		call_compute_force(0, nbodies);
		Harness::parallel_for(call_compute_force, 0, nbodies);
		Harness::stop_timing();

#ifdef TRACK_TRAVERSALS
		uint64_t sum_nodes_traversed=0;
		for (int i=0; i<nbodies; i ++ ) {
			Point *point = points_sorted[i];
			sum_nodes_traversed += point->num_nodes_traversed;
			//printf("%d %d\n", point->id, point->num_nodes_traversed);
			if(point->id==0)
				printf("position: %f %f %f %f %f %f\n", point->cofm.x, point->cofm.y, point->cofm.z, point->acc.x, point->acc.y, point->acc.z);
		}
		printf("sum_nodes_traversed:%llu\n", sum_nodes_traversed);
#endif
		for (int i=0; i<nbodies; i ++ ) {
			advance_point(points_sorted[i], half_dtime, dtime);
		}

		//printf("point 0 position: %f %f %f\n", points_sorted[0]->cofm.x, points_sorted[0]->cofm.y, points_sorted[0]->cofm.z);
#ifdef METRICS
		printLoadDistribution();
		subtrees.clear();
		max_depth = 0;
#endif
		/* Cleanup before the next run */
		free_tree(g_root);
		g_root=0;
		delete [] points_sorted;
	}

	delete [] points;

	return 0;
}

#ifdef METRICS
void printLoadDistribution()
{
	printf("num bottom subtrees %d\n",subtrees.size());
	std::vector<Node*>::iterator iter = subtrees.begin();
	for(;iter != subtrees.end();iter++)
	{
		long int num_vertices=0, footprint=0;
		int DOR=-1;
		if((*iter)->depth == 0)
		{
			DOR=0;
		}
		subtreeStats stats;
		getSubtreeStats(*iter, &stats,DOR);
		printf("id %p num_vertices %d footprint %ld\n",*iter, stats.numnodes, stats.footprint);
	}
}

void getSubtreeStats(Node* ver, subtreeStats* stats, int DOR)
{
		if(DOR == splice_depth)
		{
			return;
		}
		stats->numnodes += 1;
		stats->footprint += ver->numpointsvisited;
		assert(ver != NULL);
		if(DOR >=0)
			DOR +=1;
		for (int i = 0; i < 8; i++) {
		Node *child = ver->getChild(i);
		if (child != NULL) {
			getSubtreeStats(child,stats, DOR);
		}
		}
}


#endif
