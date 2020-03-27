/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdint.h>
#include <string>
#include <string.h>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <queue>
#include <limits>
#include <vector>

#include "harness.h"
using namespace std;

int DIM;
long int correct_cnt = 0;

class Point {
public:
	float *coord;
	int label;
	float tau;
	int closest_label;

	Point() {
		coord = new float[DIM];
#ifdef TRACK_TRAVERSALS
		num_nodes_traversed = 0;
		id = next_id++;
#endif
	}

	~Point() {
		delete coord;
	}
#ifdef TRACK_TRAVERSALS
	int num_nodes_traversed;
	int id;
	static int next_id;
#endif
};

std::vector<Point*> points, searchpoints;
#ifdef TRACK_TRAVERSALS
int Point::next_id = 0;
#endif

// to avoid conflict with std::distance :/
float mydistance(const Point* a, const Point* b) {
	float d = 0;
	for(int i = 0; i < DIM; i++) {
		float diff = a->coord[i] - b->coord[i];
		d += diff * diff;
	}
	return sqrt(d);
}

std::vector<Point*> _items;

struct Node
{
	int index;
	float threshold;
	Node *parent;
	Node* left;
	Node* right;
#ifdef METRICS
	int numPointsVisited;
#endif

#ifdef TRACK_TRAVERSALS
	int id;
	static int next_id;
#endif

	Node() :
		index(0), threshold(0.), parent(0), left(0), right(0) {
#ifdef TRACK_TRAVERSALS
		id = next_id++;
#endif
#ifdef METRICS
		numPointsVisited = 0;
#endif
	}

	~Node() {
		delete left;
		delete right;
	}
};

#ifdef TRACK_TRAVERSALS
int Node::next_id = 0;
#endif

#ifdef METRICS
std::vector<Node*> subtrees;
class subtreeStats
{
public:
	long int footprint;
	int numnodes;
	subtreeStats(){footprint=0;numnodes=0;}
};
int splice_depth=10;
void getSubtreeStats(Node* ver, subtreeStats* stat);
void printLoadDistribution();
#endif

Node* _root;

#pragma afterClassDecs

struct DistanceComparator
{
	const Point* item;
	DistanceComparator( const Point* item ) : item(item) {}
	bool operator()(const Point* a, const Point* b) {
		return mydistance( item, a ) < mydistance( item, b );
	}
};



Node* buildFromPoints( int lower, int upper, int depth ) {
	if ( upper == lower ) {
		return NULL;
	}

	Node* node = new Node();
	node->index = lower;

#ifdef METRICS
	if(depth == splice_depth-1)
		subtrees.push_back(node);
#endif
	if ( upper - lower > 1 ) {

		// choose an arbitrary point and move it to the start
		//int i = (int)((float)rand() / RAND_MAX * (upper - lower - 1) ) + lower;
		int i = lower;
		std::swap( _items[lower], _items[i] );

		int median = ( upper + lower ) / 2;

		// partitian around the median distance
		std::nth_element(
				_items.begin() + lower + 1,
				_items.begin() + median,
				_items.begin() + upper,
				DistanceComparator( _items[lower] ));

		// what was the median?
		node->threshold = mydistance( _items[lower], _items[median] );

		node->index = lower;
		node->left = buildFromPoints( lower + 1, median, depth+1 );
		if (node->left != NULL) node->left->parent = node;
		node->right = buildFromPoints( median, upper, depth+1 );
		if (node->right != NULL) node->right->parent = node;
	}

	return node;
}


void search( Node* node, Point* target)
{
	if ( node == NULL ) return;
#ifdef TRACK_TRAVERSALS
	target->num_nodes_traversed++;
#endif
#ifdef METRICS
	node->numPointsVisited++;
#endif
	if (node->parent != NULL) {
		float upperDist = mydistance( _items[node->parent->index], target );
		if (node->parent->right == node) {
			if (upperDist + target->tau < node->parent->threshold) return;
		} else if (node->parent->left == node) {
			if (upperDist - target->tau > node->parent->threshold) return;
		}
	}
	float dist = mydistance( _items[node->index], target );
	//printf("dist=%g tau=%gn", dist, target->tau );

	if ( dist < target->tau ) {
		target->closest_label = _items[node->index]->label;
		target->tau = dist;
	}

	if ( node->left == NULL && node->right == NULL ) {
		return;
	}

	if ( dist < node->threshold ) {
		search( node->left, target);
		search( node->right, target);
	} else {
		search( node->right, target);
		search( node->left, target);
	}
}

void create( const std::vector<Point*>& items ) {
	delete _root;
	_items = items;
	_root = buildFromPoints(0, items.size(), 0);
}

void search_entry( Point* target)
{
	target->tau = std::numeric_limits<float>::max();
	search( _root, target);
}

void find_nearest_neighbors(int start, int end) {
	
	for (int i = start; i < end; i++) {
		Point* target = searchpoints[i];
		search_entry( target);
	}
}

Point *read_point(FILE *in, int label) {
	Point *p = new Point;
	/*if(fscanf(in, "%d", &p->label) != 1) {
		fprintf(stderr, "Input file not large enough.\n");
		exit(1);
	}*/
	for(int j = 0; j < DIM; j++) {
		if(fscanf(in, "%f", &p->coord[j]) != 1) {
			fprintf(stderr, "Input file not large enough.\n");
			exit(1);
		}
	}
	p->id=label;
	return p;
}

Point *gen_point() {
	Point *p = new Point;
	for (int j = 0; j < DIM; j++) {
		p->coord[j] = (float)rand() / RAND_MAX;
	}
	return p;
}

enum {
	//Arg_binary,
	Arg_dim,
	Arg_K,
	Arg_npoints,
	Arg_nsearchpoints,
	Arg_inputfile,
	Arg_num,
};

void read_input(int argc, char **argv, std::vector<Point*>& points, std::vector<Point*>& searchpoints) {
	if(argc != Arg_num && argc != Arg_num - 1) {
		fprintf(stderr, "usage: vptree <dim> <K> <npoints> <nsearchpoints> [input_file]\n");
		exit(1);
	}
	DIM = atoi(argv[Arg_dim]);
	int npoints = atol(argv[Arg_npoints]);
	if (npoints <= 0) {
		fprintf(stderr, "Not enough points.\n");
		exit(1);
	}

	int nsearchpoints = atol(argv[Arg_nsearchpoints]);
	if (nsearchpoints <= 0) {
		fprintf(stderr, "Not enough points.\n");
		exit(1);
	}
	if (argc == Arg_num) {
		char *filename = argv[Arg_inputfile];
		FILE *in = fopen(filename, "r");
		if( in == NULL) {
			fprintf(stderr, "Could not open %s\n", filename);
			exit(1);
		}

		for (int i = 0; i < npoints; i++) {
			points.push_back(read_point(in,i));
		}
		for (int i = 0; i < nsearchpoints; i++) {
			searchpoints.push_back(read_point(in,i));
			//searchpoints.push_back(points[i]);
		}
		fclose(in);
	} else {
		for (int i = 0; i < npoints; i++) {
			points.push_back(gen_point());
		}
		for (int i = 0; i < npoints; i++) {
			searchpoints.push_back(gen_point());
		}
	}
}


struct SortComparator
{
	const int split;
	SortComparator( const int split ) : split(split) {}
	bool operator()(const Point* a, const Point* b) {
		return a->coord[split] < b->coord[split];
	}
};

void sort_points(std::vector<Point*>& points, int lb, int ub, int depth) {
	int size = ub - lb + 1;
	if (size <= 4) {
		return;
	}

	int split = depth % DIM;
	std::sort (points.begin() + lb, points.begin() + ub + 1, SortComparator(split));
	int mid = (ub + lb) / 2;

	sort_points(points, lb, mid, depth + 1);
	sort_points(points, mid+1, ub, depth + 1);
}

int app_main( int argc, char* argv[] ) {
	srand(0);
	read_input(argc, argv, points, searchpoints);

	create( points );

	int nsearchpoints = searchpoints.size();

	if (Harness::get_sort_flag()) {
		sort_points(searchpoints, 0, nsearchpoints - 1, 0);
	}

	Harness::start_timing();
	double startTime = clock();
	Harness::parallel_for(find_nearest_neighbors, 0, nsearchpoints);
	double endTime = clock();
	Harness::stop_timing();

	printf("time consumed %f\n",(endTime-startTime)/CLOCKS_PER_SEC);
#ifdef TRACK_TRAVERSALS
	long int sum_nodes_traversed = 0;
	for (int i = 0; i < nsearchpoints; i++) {
		Point* p = searchpoints[i];
		sum_nodes_traversed += p->num_nodes_traversed;
		if (p->label == p->closest_label) {
			correct_cnt++;
		}
		//printf("%d %d\n", p->id, p->num_nodes_traversed);
	}
	printf("sum_nodes_traversed:%ld\n", sum_nodes_traversed);
#endif

	float correct_rate = (float) correct_cnt / nsearchpoints;
	printf("correct rate: %.4f\n", correct_rate);

#ifdef METRICS
	printLoadDistribution();
#endif
	delete _root;
	_root = NULL;

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
		subtreeStats stats;
		getSubtreeStats(*iter, &stats);
		printf("(%p) num_vertices %d footprint %ld\n",*iter, stats.numnodes, stats.footprint);
	}
}

void getSubtreeStats(Node* ver, subtreeStats* stats)
{
		stats->numnodes += 1;
		stats->footprint += ver->numPointsVisited;
		assert(ver != NULL);

		if((ver->left == NULL) && (ver->right==NULL))
		{
			return;
		}

		if(ver->left)
		{
			getSubtreeStats(ver->left,stats);
		}
		if(ver->right)
		{
			getSubtreeStats(ver->right,stats);
		}
}


#endif
