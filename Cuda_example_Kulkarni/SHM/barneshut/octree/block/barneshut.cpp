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

#define __stderrp stderr
#define __stdinp stdin

class Vec 
{
public: float x;
float y;
float z;


inline Vec()
{
}


inline Vec(float f)
{
	(this) -> x = f;
	(this) -> y = f;
	(this) -> z = f;
}
}
;
enum bh_node_type {bhLeafNode,bhNonLeafNode};

class Point 
{


public: inline Point()
{
#ifdef TRACK_TRAVERSALS
	(this) -> id = next_id++;
#endif
}
float mass;
/* center of mass */
class Vec cofm;
/* current velocity */
class Vec vel;
/* current acceleration */
class Vec acc;
#ifdef TRACK_TRAVERSALS
int num_nodes_traversed;
int id;
private: static int next_id;
#endif
}
;

class Node 
{


public: inline Node(class Point *p) : type(bhLeafNode), point(p)
{
	(this) ->  clearChildren ();
}


inline Node() : type(bhNonLeafNode), point(::new Point )
{
	(this) ->  clearChildren ();
}


inline ~Node()
{
	if (((this) -> type) == bhNonLeafNode)
		:: delete ((this) -> point);
}
Node *getChild(int i);
void setChild(int i,class Node *c);
enum bh_node_type type;
class Point *point;
class Node *child0;
class Node *child1;
class Node *child2;
class Node *child3;
class Node *child4;
class Node *child5;
class Node *child6;
class Node *child7;
private: void clearChildren();
}
;
#ifdef TRACK_TRAVERSALS
int Point::next_id = 0;
#endif

void Node::clearChildren()
{
	(this) -> child0 = 0L;
	(this) -> child1 = 0L;
	(this) -> child2 = 0L;
	(this) -> child3 = 0L;
	(this) -> child4 = 0L;
	(this) -> child5 = 0L;
	(this) -> child6 = 0L;
	(this) -> child7 = 0L;
}

Node *Node::getChild(int i)
{
	if (i == 0)
		return (this) -> child0;
	else if (i == 1)
		return (this) -> child1;
	else if (i == 2)
		return (this) -> child2;
	else if (i == 3)
		return (this) -> child3;
	else if (i == 4)
		return (this) -> child4;
	else if (i == 5)
		return (this) -> child5;
	else if (i == 6)
		return (this) -> child6;
	else
		return (this) -> child7;
}

void Node::setChild(int i,class Node *c)
{
	if (i == 0)
		(this) -> child0 = c;
	else if (i == 1)
		(this) -> child1 = c;
	else if (i == 2)
		(this) -> child2 = c;
	else if (i == 3)
		(this) -> child3 = c;
	else if (i == 4)
		(this) -> child4 = c;
	else if (i == 5)
		(this) -> child5 = c;
	else if (i == 6)
		(this) -> child6 = c;
	else
		(this) -> child7 = c;
}
#include "block.h"
#include "interstate.h"
#include "autotuner.h"

#pragma afterClassDecs

#ifdef BLOCK_PROFILE
#include "blockprofiler.h"
BlockProfiler profiler;
#endif
#ifdef PARALLELISM_PROFILE
#include "parallelismprofiler.h"
ParallelismProfiler *parallelismProfiler;
#endif

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
class Node *g_root;
class Point **points_sorted;
int global_timestep;
enum __unnamed_enum___F0_L125_C1_Arg_binary__COMMA__Arg_inputfile__COMMA__Arg_npoints__COMMA__Arg_num {
	//Arg_binary,
	Arg_inputfile,Arg_npoints,Arg_num};

class Point *read_input(int argc,char **argv)
{
	if (argc != Arg_num) {
		fprintf(__stderrp,"Usage: [input_file] [nbodies]\n");
		exit(1);
	}
	FILE *infile = fopen(argv[Arg_inputfile],"r");
	if (!infile) {
		fprintf(__stderrp,"Error: could not read input file: %s\n",argv[Arg_inputfile]);
		exit(1);
	}
	nbodies = 0;
	nbodies = (atoll(argv[Arg_npoints]));
	if (nbodies <= 0) {
		fprintf(__stderrp,"Error: nbodies not valid.\n");
		exit(1);
	}
	printf("Overriding nbodies from input file. nbodies = %lld\n",nbodies);
	uint64_t junk;
	fscanf(infile,"%lld",&junk);
	if (nbodies <= 0) {
		fscanf(infile,"%lld",&nbodies);
		if (nbodies < 1) {
			fprintf(__stderrp,"Error: nbodies must be at least 1!\n");
			exit(1);
		}
	}
	fscanf(infile,"%lld",&ntimesteps);
	if (ntimesteps < 1) {
		fprintf(__stderrp,"Error: ntimesteps must be at least 1!\n");
		exit(1);
	}
	fscanf(infile,"%f",&dtime);
	if (dtime <= 0.0) {
		fprintf(__stderrp,"Error: dtime can not be zero!\n");
		exit(1);
	}
	fscanf(infile,"%f",&eps);
	fscanf(infile,"%f",&tol);
	half_dtime = (0.5 * dtime);
	inv_tol_squared = (1.0 / (tol * tol));
	eps_squared = (eps * eps);
	class Point *points = new Point [nbodies];
	for (int i = 0; i < nbodies; i++) {
		int ret = fscanf(infile,"%f %f %f %f %f %f %f",&points[i].Point::mass,&points[i].Point::cofm.Vec::x,&points[i].Point::cofm.Vec::y,&points[i].Point::cofm.Vec::z,&points[i].Point::vel.Vec::x,&points[i].Point::vel.Vec::y,&points[i].Point::vel.Vec::z);
		if (ret != 7) {
			fprintf(__stderrp,"Error: Invalid point (%d).\n",i);
			exit(1);
		}
		points[i].Point::acc.Vec::x = (points[i].Point::acc.Vec::y = (points[i].Point::acc.Vec::z = 0.0));
		points[i].id=i;
#ifdef TRACK_TRAVERSALS
		points[i].Point::num_nodes_traversed = 0;
#endif
	}
	if (infile != __stdinp) {
		fclose(infile);
	}
	return points;
}

void compute_bounding_box(class Point *points,class Vec *center)
{
	class Vec pos;
	class Vec min(3.40282347e+38F);
	class Vec max(1.17549435e-38F);
	/* compute the max and min positions to form a bounding box */
	for (int i = 0; i < nbodies; i++) {
		pos = points[i].Point::cofm;
		if (min.Vec::x > pos.Vec::x) {
			min.Vec::x = pos.Vec::x;
		}
		if (min.Vec::y > pos.Vec::y) {
			min.Vec::y = pos.Vec::y;
		}
		if (min.Vec::z > pos.Vec::z) {
			min.Vec::z = pos.Vec::z;
		}
		if (max.Vec::x < pos.Vec::x) {
			max.Vec::x = pos.Vec::x;
		}
		if (max.Vec::y < pos.Vec::y) {
			max.Vec::y = pos.Vec::y;
		}
		if (max.Vec::z < pos.Vec::z) {
			max.Vec::z = pos.Vec::z;
		}
	}
	/* compute the maximum of the diameters of all axes */
	diameter = (max.Vec::x - min.Vec::x);
	if (diameter < (max.Vec::y - min.Vec::y)) {
		diameter = (max.Vec::y - min.Vec::y);
	}
	if (diameter < (max.Vec::z - min.Vec::z)) {
		diameter = (max.Vec::z - min.Vec::z);
	}
	/* compute the center point */
	center -> Vec::x = ((max.Vec::x + min.Vec::x) * 0.5);
	center -> Vec::y = ((max.Vec::y + min.Vec::y) * 0.5);
	center -> Vec::z = ((max.Vec::z + min.Vec::z) * 0.5);
}

void insert_point(class Node *root,class Point *p,float r,int depth)
{
	class Vec offset(0.0f);
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
	int space = 0;
	class Point *rootPoint = (root -> Node::point);
	if (rootPoint -> Point::cofm.Vec::x < p -> Point::cofm.Vec::x) {
		space += 1;
		offset.Vec::x = r;
	}
	if (rootPoint -> Point::cofm.Vec::y < p -> Point::cofm.Vec::y) {
		space += 2;
		offset.Vec::y = r;
	}
	if (rootPoint -> Point::cofm.Vec::z < p -> Point::cofm.Vec::z) {
		space += 4;
		offset.Vec::z = r;
	}
	class Node *child = root ->  getChild (space);
	if (child == 0L) {
		class Node *n = ::new Node (p);
		root ->  setChild (space,n);
	}
	else {
		float half_r = (0.5F * r);
		if ((child -> Node::type) == bhLeafNode) {
			class Node *newNode = ::new Node ;
			class Point *nodePoint = (newNode -> Node::point);
			nodePoint -> Point::cofm.Vec::x = ((rootPoint -> Point::cofm.Vec::x - half_r) + offset.Vec::x);
			nodePoint -> Point::cofm.Vec::y = ((rootPoint -> Point::cofm.Vec::y - half_r) + offset.Vec::y);
			nodePoint -> Point::cofm.Vec::z = ((rootPoint -> Point::cofm.Vec::z - half_r) + offset.Vec::z);
			insert_point(newNode,p,half_r,(depth + 1));
			insert_point(newNode,(child -> Node::point),half_r,(depth + 1));
			root ->  setChild (space,newNode);
		}
		else {
			insert_point(child,p,half_r,(depth + 1));
		}
	}
}

void free_tree(class Node *root)
{
	for (int i = 0; i < 8; i++) {
		class Node *child = root ->  getChild (i);
		if ((child != 0L) && ((child -> Node::type) == bhNonLeafNode)) {
			free_tree(child);
			root ->  setChild (i,0L);
		}
	}
	delete root;
}

void compute_center_of_mass(class Node *node)
{
	int j = 0;
	class Vec cofm(0.0f);
	float mass = 0.0;
	for (int i = 0; i < 8; i++) {
		class Node *child = node ->  getChild (i);
		if (child != 0L) {
			/* compact child nodes for speed */
			if (i != j) {
				node ->  setChild (j,node ->  getChild (i));
				node ->  setChild (i,0L);
			}
			j++;
			/* If non leave node need to traverse children: */
			if ((child -> Node::type) == bhNonLeafNode) {
				/* summarize children */
				compute_center_of_mass(child);
			}
			else {
				/* insert this point in sorted order */
				points_sorted[sortidx++] = (child -> Node::point);
			}
			class Point *childPoint = (child -> Node::point);
			mass += (childPoint -> Point::mass);
			cofm.Vec::x += (childPoint -> Point::cofm.Vec::x * (childPoint -> Point::mass));
			cofm.Vec::y += (childPoint -> Point::cofm.Vec::y * (childPoint -> Point::mass));
			cofm.Vec::z += (childPoint -> Point::cofm.Vec::z * (childPoint -> Point::mass));
		}
	}
	class Point *nodePoint = (node -> Node::point);
	nodePoint -> Point::cofm.Vec::x = (cofm.Vec::x / mass);
	nodePoint -> Point::cofm.Vec::y = (cofm.Vec::y / mass);
	nodePoint -> Point::cofm.Vec::z = (cofm.Vec::z / mass);
	nodePoint -> Point::mass = mass;
}

inline void update_point(class Point *p,class Node *root,const class Vec &dr,float drsq,float epssq)
{
	drsq += epssq;
	float idr = (1.0 / sqrt(drsq));
	float nphi = (( *(root -> Node::point)).Point::mass * idr);
	float scale = ((nphi * idr) * idr);
	p -> Point::acc.Vec::x += (dr.Vec::x * scale);
	p -> Point::acc.Vec::y += (dr.Vec::y * scale);
	p -> Point::acc.Vec::z += (dr.Vec::z * scale);
}

void compute_force_recursive_mapRoot(class Node *node,float dsq,float epssq)
{
	_Block::root_node = node;
	_Block::root_dsq = dsq;
	_Block::root_epssq = epssq;
}

void compute_force_recursive_block(class Node *node,float dsq,float epssq,class _BlockStack *_stack,int _depth)
{
	class _BlockSet *_set = _stack ->  get (_depth);
	class _Block *_block = _set -> block;
	class _Block *_nextBlock0 = &_set -> _BlockSet::nextBlock0;
	_nextBlock0 ->  recycle ();
#ifdef BLOCK_PROFILE
	profiler.record(_block->size);
#endif

	for (int _bi = 0; _bi < _block -> _Block::size; ++_bi) {
		class _Point &_point = _block ->  get (_bi);
		class Point *p = _point._Point::p;
        if (node == 0L) {
#ifdef PARALLELISM_PROFILE
        	parallelismProfiler->recordTruncate();
#endif
            continue;
        }
#ifdef TRACK_TRAVERSALS	
		p -> Point::num_nodes_traversed++;
#endif
		class Vec dr;
		class Point *nodePoint = (node -> Node::point);
		dr.Vec::x = (nodePoint -> Point::cofm.Vec::x - p -> Point::cofm.Vec::x);
		dr.Vec::y = (nodePoint -> Point::cofm.Vec::y - p -> Point::cofm.Vec::y);
		dr.Vec::z = (nodePoint -> Point::cofm.Vec::z - p -> Point::cofm.Vec::z);
		float drsq = (((dr.Vec::x * dr.Vec::x) + (dr.Vec::y * dr.Vec::y)) + (dr.Vec::z * dr.Vec::z));
		/*
    always do recursive to get most exact
    particle is far away can stop here
		 */
		if (drsq < dsq) {
			if ((node -> Node::type) == bhNonLeafNode) {
				_nextBlock0 ->  add (p);
#ifdef PARALLELISM_PROFILE
	parallelismProfiler->recordRecurse();
#endif
			} else {
				/* two different particles */
				if (p != (node -> Node::point)) {
					update_point(p,node,dr,drsq,epssq);
				}
#ifdef PARALLELISM_PROFILE
	parallelismProfiler->recordTruncate();
#endif
			}
		}
		else {
			drsq += epssq;
			update_point(p,node,dr,drsq,epssq);
#ifdef PARALLELISM_PROFILE
	parallelismProfiler->recordTruncate();
#endif
		}
	}
	if (_nextBlock0 -> _Block::size > 0) {
		_stack ->  get (_depth + 1) -> _BlockSet::block = _nextBlock0;
		{
			dsq *= 0.25;
			compute_force_recursive_block((node -> Node::child0),dsq,epssq,_stack,_depth + 1);
			compute_force_recursive_block((node -> Node::child1),dsq,epssq,_stack,_depth + 1);
			compute_force_recursive_block((node -> Node::child2),dsq,epssq,_stack,_depth + 1);
			compute_force_recursive_block((node -> Node::child3),dsq,epssq,_stack,_depth + 1);
			compute_force_recursive_block((node -> Node::child4),dsq,epssq,_stack,_depth + 1);
			compute_force_recursive_block((node -> Node::child5),dsq,epssq,_stack,_depth + 1);
			compute_force_recursive_block((node -> Node::child6),dsq,epssq,_stack,_depth + 1);
			compute_force_recursive_block((node -> Node::child7),dsq,epssq,_stack,_depth + 1);
		}
	}
#ifdef PARALLELISM_PROFILE
	parallelismProfiler->blockEnd();
#endif
}

void compute_force_recursive_blockAutotune(class Node *node,float dsq,float epssq,class _BlockStack *_stack,int _depth, _Autotuner *_autotuner)
{
	class _BlockSet *_set = _stack ->  get (_depth);
	class _Block *_block = _set -> block;
	class _Block *_nextBlock0 = &_set -> _BlockSet::nextBlock0;
	_autotuner->profileWorkDone(_block->size);
	_nextBlock0 ->  recycle ();
#ifdef BLOCK_PROFILE
	profiler.record(_block->size);
#endif

	for (int _bi = 0; _bi < _block -> _Block::size; ++_bi) {
		class _Point &_point = _block ->  get (_bi);
		class Point *p = _point._Point::p;
		if (node == 0L) {
			continue;
		}
#ifdef TRACK_TRAVERSALS
		p -> Point::num_nodes_traversed++;
#endif
		class Vec dr;
		class Point *nodePoint = (node -> Node::point);
		dr.Vec::x = (nodePoint -> Point::cofm.Vec::x - p -> Point::cofm.Vec::x);
		dr.Vec::y = (nodePoint -> Point::cofm.Vec::y - p -> Point::cofm.Vec::y);
		dr.Vec::z = (nodePoint -> Point::cofm.Vec::z - p -> Point::cofm.Vec::z);
		float drsq = (((dr.Vec::x * dr.Vec::x) + (dr.Vec::y * dr.Vec::y)) + (dr.Vec::z * dr.Vec::z));
		/*
    always do recursive to get most exact
    particle is far away can stop here
		 */
		if (drsq < dsq) {
			if ((node -> Node::type) == bhNonLeafNode) {
				_nextBlock0 ->  add (p);
			}
			else {
				if (p != (node -> Node::point)) {
					update_point(p,node,dr,drsq,epssq);
				}
				/* two different particles */
			}
		}
		else {
			drsq += epssq;
			update_point(p,node,dr,drsq,epssq);
		}
	}
	if (_nextBlock0 -> _Block::size > 0) {
		_stack ->  get (_depth + 1) -> _BlockSet::block = _nextBlock0;
		{
			dsq *= 0.25;
			compute_force_recursive_blockAutotune((node -> Node::child0),dsq,epssq,_stack,_depth + 1, _autotuner);
			compute_force_recursive_blockAutotune((node -> Node::child1),dsq,epssq,_stack,_depth + 1, _autotuner);
			compute_force_recursive_blockAutotune((node -> Node::child2),dsq,epssq,_stack,_depth + 1, _autotuner);
			compute_force_recursive_blockAutotune((node -> Node::child3),dsq,epssq,_stack,_depth + 1, _autotuner);
			compute_force_recursive_blockAutotune((node -> Node::child4),dsq,epssq,_stack,_depth + 1, _autotuner);
			compute_force_recursive_blockAutotune((node -> Node::child5),dsq,epssq,_stack,_depth + 1, _autotuner);
			compute_force_recursive_blockAutotune((node -> Node::child6),dsq,epssq,_stack,_depth + 1, _autotuner);
			compute_force_recursive_blockAutotune((node -> Node::child7),dsq,epssq,_stack,_depth + 1, _autotuner);
		}
	}
}

void compute_force_recursive(class Point *p,class Node *node,float dsq,float epssq)
{
	if (node == 0L)
		return ;
#ifdef TRACK_TRAVERSALS	
	p -> Point::num_nodes_traversed++;
#endif
	class Vec dr;
	class Point *nodePoint = (node -> Node::point);
	dr.Vec::x = (nodePoint -> Point::cofm.Vec::x - p -> Point::cofm.Vec::x);
	dr.Vec::y = (nodePoint -> Point::cofm.Vec::y - p -> Point::cofm.Vec::y);
	dr.Vec::z = (nodePoint -> Point::cofm.Vec::z - p -> Point::cofm.Vec::z);
	float drsq = (((dr.Vec::x * dr.Vec::x) + (dr.Vec::y * dr.Vec::y)) + (dr.Vec::z * dr.Vec::z));
	/*
    always do recursive to get most exact
    particle is far away can stop here
	 */
	if (drsq < dsq) {
		if ((node -> Node::type) == bhNonLeafNode) {
			dsq *= 0.25;
			compute_force_recursive(p,(node -> Node::child0),dsq,epssq);
			compute_force_recursive(p,(node -> Node::child1),dsq,epssq);
			compute_force_recursive(p,(node -> Node::child2),dsq,epssq);
			compute_force_recursive(p,(node -> Node::child3),dsq,epssq);
			compute_force_recursive(p,(node -> Node::child4),dsq,epssq);
			compute_force_recursive(p,(node -> Node::child5),dsq,epssq);
			compute_force_recursive(p,(node -> Node::child6),dsq,epssq);
			compute_force_recursive(p,(node -> Node::child7),dsq,epssq);
		}
		else {
			/* two different particles */
			if (p != (node -> Node::point)) {
				update_point(p,node,dr,drsq,epssq);
			}
		}
	}
	else {
		drsq += epssq;
		update_point(p,node,dr,drsq,epssq);
	}
}

void advance_point(class Point *p,float dthf,float dtime)
{
	class Vec delta_v;
	class Vec velh;
	delta_v.Vec::x = (p -> Point::acc.Vec::x * dthf);
	delta_v.Vec::y = (p -> Point::acc.Vec::y * dthf);
	delta_v.Vec::z = (p -> Point::acc.Vec::z * dthf);
	velh.Vec::x = (p -> Point::vel.Vec::x + delta_v.Vec::x);
	velh.Vec::y = (p -> Point::vel.Vec::y + delta_v.Vec::y);
	velh.Vec::z = (p -> Point::vel.Vec::z + delta_v.Vec::z);
	p -> Point::cofm.Vec::x += (velh.Vec::x * dtime);
	p -> Point::cofm.Vec::y += (velh.Vec::y * dtime);
	p -> Point::cofm.Vec::z += (velh.Vec::z * dtime);
	p -> Point::vel.Vec::x = (velh.Vec::x + delta_v.Vec::x);
	p -> Point::vel.Vec::y = (velh.Vec::y + delta_v.Vec::y);
	p -> Point::vel.Vec::z = (velh.Vec::z + delta_v.Vec::z);
}

void compute_force(int start,int end,class Point **points,class Node *root,float size,float itolsq,int step,float dthf,float epssq)
{
	float dsq = ((size * size) * itolsq);

#pragma parallelForOnTree
	if (Harness::get_block_size() == 0) {
		compute_force_recursive_mapRoot(root,dsq,epssq);

		_Autotuner _autotuner(end);
		_BlockStack *_tuneStack = new _BlockStack();
		_Block *_tuneBlock = new _Block();
		_IntermediaryBlock *_tuneInterBlock = new _IntermediaryBlock();
		_tuneInterBlock->block = _tuneBlock;

		int **_tuneIndexes = _autotuner.tune();
		for (int _t = 0; _t < _autotuner.tuneIndexesCnt; _t++) {
			int *_indexes = _tuneIndexes[_t];
			_autotuner.tuneEntryBlock();
			//cout << _autotuner.sampleSizes[_t] << endl;
			_tuneInterBlock->reset();
			for(int _tt = 0; _tt < _autotuner.sampleSizes[_t]; _tt++) {
				int i = _indexes[_tt];
				class Point *p = points[i];
				class Vec a_prev;
				a_prev = (p -> Point::acc);
				p -> Point::acc.Vec::x = (p -> Point::acc.Vec::y = (p -> Point::acc.Vec::z = 0.0));
				_tuneBlock ->  add (p);
				struct _IntermediaryState *_interState = _tuneInterBlock->next ();
				_interState -> _IntermediaryState::p = p;
				_interState -> _IntermediaryState::a_prev = a_prev;
			}
			_tuneStack ->  get (0) -> block = _tuneBlock;
			compute_force_recursive_blockAutotune(_Block::root_node,_Block::root_dsq,_Block::root_epssq,_tuneStack,0,&_autotuner);
			_tuneBlock ->  recycle ();
			_tuneInterBlock->reset();
			for(int _tt = 0; _tt < _autotuner.sampleSizes[_t]; _tt++) {
				struct _IntermediaryState *_interState = _tuneInterBlock->next ();
				class Point *p = _interState -> _IntermediaryState::p;
				class Vec a_prev = _interState -> _IntermediaryState::a_prev;
				if (step > 0) {
					class Vec delta_v;
					delta_v.Vec::x = ((p -> Point::acc.Vec::x - a_prev.Vec::x) * dthf);
					delta_v.Vec::y = ((p -> Point::acc.Vec::y - a_prev.Vec::y) * dthf);
					delta_v.Vec::z = ((p -> Point::acc.Vec::z - a_prev.Vec::z) * dthf);
					p -> Point::vel.Vec::x += delta_v.Vec::x;
					p -> Point::vel.Vec::y += delta_v.Vec::y;
					p -> Point::vel.Vec::z += delta_v.Vec::z;
				}
			}
			_autotuner.tuneExitBlock(_t);
		}
		_autotuner.tuneFinished();
		delete _tuneStack;
		delete _tuneBlock;
		delete _tuneInterBlock;


		class _BlockStack *_stack = new _BlockStack;
		class _Block *_block = new _Block;
		class _IntermediaryBlock *_interBlock = new _IntermediaryBlock;
		_interBlock->_IntermediaryBlock::block = _block;
		for (int _start = start; _start < end; _start += _Block::max_block) {
			int _end = min(_start + _Block::max_block,end);
			_interBlock-> reset ();
			for (int i = _start; i < _end; ++i) {
				if(_autotuner.isSampled(i)) continue ;
				class Point *p = points[i];
				class Vec a_prev;
				a_prev = (p -> Point::acc);
				p -> Point::acc.Vec::x = (p -> Point::acc.Vec::y = (p -> Point::acc.Vec::z = 0.0));
				_block->add (p);
				struct _IntermediaryState *_interState = _interBlock-> next ();
				_interState -> _IntermediaryState::p = p;
				_interState -> _IntermediaryState::a_prev = a_prev;
			}
			_stack-> get (0) -> block = _block;
			compute_force_recursive_block(_Block::root_node,_Block::root_dsq,_Block::root_epssq,_stack,0);
			_block-> recycle ();
			_interBlock-> reset ();
			for (int i = _start; i < _end; ++i) {
				if(_autotuner.isSampled(i)) continue ;
				struct _IntermediaryState *_interState = _interBlock-> next ();
				class Point *p = _interState -> _IntermediaryState::p;
				class Vec a_prev = _interState -> _IntermediaryState::a_prev;
				if (step > 0) {
					class Vec delta_v;
					delta_v.Vec::x = ((p -> Point::acc.Vec::x - a_prev.Vec::x) * dthf);
					delta_v.Vec::y = ((p -> Point::acc.Vec::y - a_prev.Vec::y) * dthf);
					delta_v.Vec::z = ((p -> Point::acc.Vec::z - a_prev.Vec::z) * dthf);
					p -> Point::vel.Vec::x += delta_v.Vec::x;
					p -> Point::vel.Vec::y += delta_v.Vec::y;
					p -> Point::vel.Vec::z += delta_v.Vec::z;
				}
			}
		}

	} else {

		compute_force_recursive_mapRoot(root,dsq,epssq);
		class _BlockStack _stack;
		class _Block _block;
		class _IntermediaryBlock _interBlock;
		_interBlock._IntermediaryBlock::block = &_block;
		for (int _start = start; _start < end; _start += _Block::max_block) {
			int _end = min(_start + _Block::max_block,end);
			_interBlock. reset ();
			for (int i = _start; i < _end; ++i) {
				class Point *p = points[i];
				class Vec a_prev;
				a_prev = (p -> Point::acc);
				p -> Point::acc.Vec::x = (p -> Point::acc.Vec::y = (p -> Point::acc.Vec::z = 0.0));
				_block. add (p);
				struct _IntermediaryState *_interState = _interBlock. next ();
				_interState -> _IntermediaryState::p = p;
				_interState -> _IntermediaryState::a_prev = a_prev;
			}
			_stack. get (0) -> block = &_block;
			compute_force_recursive_block(_Block::root_node,_Block::root_dsq,_Block::root_epssq,&_stack,0);
			_block. recycle ();
			_interBlock. reset ();
			for (int i = _start; i < _end; ++i) {
				struct _IntermediaryState *_interState = _interBlock. next ();
				class Point *p = _interState -> _IntermediaryState::p;
				class Vec a_prev = _interState -> _IntermediaryState::a_prev;
				if (step > 0) {
					class Vec delta_v;
					delta_v.Vec::x = ((p -> Point::acc.Vec::x - a_prev.Vec::x) * dthf);
					delta_v.Vec::y = ((p -> Point::acc.Vec::y - a_prev.Vec::y) * dthf);
					delta_v.Vec::z = ((p -> Point::acc.Vec::z - a_prev.Vec::z) * dthf);
					p -> Point::vel.Vec::x += delta_v.Vec::x;
					p -> Point::vel.Vec::y += delta_v.Vec::y;
					p -> Point::vel.Vec::z += delta_v.Vec::z;
				}
			}
		}
	}
}

void call_compute_force(int start,int end)
{
	//cout << start << " " << end << endl;
	compute_force(start,end,points_sorted,g_root,diameter,inv_tol_squared,global_timestep,half_dtime,eps_squared);
}

int app_main(int argc,char **argv)
{
	class Point *points = read_input(argc,argv);
	printf("Configuration: nbodies = %lld, ntimesteps = %lld, dtime = %f eps = %f, tol = %f\n",nbodies,ntimesteps,dtime,eps,tol);
	for (int t = 0; t < ntimesteps; t++) {
		printf("Time step %d:\n",t);
		class Vec center;
		compute_bounding_box(points,&center);
		g_root = (::new Node );
		( *(g_root -> Node::point)).Point::cofm = center;
		for (int i = 0; i < nbodies; i++) {
			insert_point(g_root,(points + i),(diameter * 0.5),0);
		}
		points_sorted = (new Point *[nbodies]);
		sortidx = 0;
		compute_center_of_mass(g_root);
		// unsort points
		if (!Harness::get_sort_flag()) {
			for (int i = 0; i < nbodies; i++) {
				points_sorted[i] = (points + i);
			}
		}
#ifdef PARALLELISM_PROFILE
	parallelismProfiler = new ParallelismProfiler;
#endif
		Harness::start_timing();
		if (Harness::get_block_size() > 0) {
			_Block::max_block = Harness::get_block_size();
			_IntermediaryBlock::max_block = Harness::get_block_size();
		}
		global_timestep = t;
		//		compute_force(0, nbodies, points_sorted, g_root, diameter, inv_tol_squared,
		//				global_timestep, half_dtime, eps_squared);
		//    call_compute_force(0,nbodies);
		Harness::parallel_for(call_compute_force, 0, nbodies);
		Harness::stop_timing();
#ifdef TRACK_TRAVERSALS
		uint64_t sum_nodes_traversed = 0;
		for (int i = 0; i < nbodies; i++) {
			class Point *point = points_sorted[i];
			sum_nodes_traversed += (point -> Point::num_nodes_traversed);
			//printf("%d %d\n", point->id, point->num_nodes_traversed);
			if(point->id == 0)
				printf("position: %f %f %f %f %f %f\n",point->cofm.x,point->cofm.y,point->cofm.z, point->acc.x,point->acc.y,point->acc.z);
		}
		printf("sum_nodes_traversed:%llu\n",sum_nodes_traversed);
#endif
#ifdef BLOCK_PROFILE
	profiler.output();
#endif
#ifdef PARALLELISM_PROFILE
	parallelismProfiler->output();
	delete parallelismProfiler;
#endif
		for (int i = 0; i < nbodies; i++) {
			advance_point(points_sorted[i],half_dtime,dtime);
		}
		/* Cleanup before the next run */
		free_tree(g_root);
		g_root = 0;
		delete []points_sorted;
	}
	delete []points;
	return 0;
}
